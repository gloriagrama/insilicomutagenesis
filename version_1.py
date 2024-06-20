#!/usr/bin/env python
# coding: utf-8


# ### imports

# In[30]:


import tensorflow as tf
import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[31]:


transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1'
fasta_file = '/home/jovyan/data4/ggrama_runAI/data/genome.fa'
clinvar_vcf = '/home/jovyan/data4/ggrama_runAI/data/clinvar.vcf.gz'


# ## enformer code

# In[32]:


SEQUENCE_LENGTH = 393216

class Enformer:

  def __init__(self, tfhub_url):
    self._model = hub.load(tfhub_url).model

  def predict_on_batch(self, inputs):
    predictions = self._model.predict_on_batch(inputs)
    return {k: v.numpy() for k, v in predictions.items()}

  @tf.function
  def contribution_input_grad(self, input_sequence,
                              target_mask, output_head='human'):
    input_sequence = input_sequence[tf.newaxis]

    target_mask_mass = tf.reduce_sum(target_mask)
    with tf.GradientTape() as tape:
      tape.watch(input_sequence)
      prediction = tf.reduce_sum(
          target_mask[tf.newaxis] *
          self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

    input_grad = tape.gradient(prediction, input_sequence) * input_sequence
    input_grad = tf.squeeze(input_grad, axis=0)
    return tf.reduce_sum(input_grad, axis=-1)


class EnformerScoreVariantsRaw:

  def __init__(self, tfhub_url, organism='human'):
    self._model = Enformer(tfhub_url)
    self._organism = organism
  
  def predict_on_batch(self, inputs):
    ref_prediction = self._model.predict_on_batch(inputs['ref'])[self._organism]
    alt_prediction = self._model.predict_on_batch(inputs['alt'])[self._organism]

    return alt_prediction.mean(axis=1) - ref_prediction.mean(axis=1)


class EnformerScoreVariantsNormalized:

  def __init__(self, tfhub_url, transform_pkl_path,
               organism='human'):
    assert organism == 'human', 'Transforms only compatible with organism=human'
    self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
    with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
      transform_pipeline = joblib.load(f)
    self._transform = transform_pipeline.steps[0][1]  # StandardScaler.
    
  def predict_on_batch(self, inputs):
    scores = self._model.predict_on_batch(inputs)
    return self._transform.transform(scores)


class EnformerScoreVariantsPCANormalized:

  def __init__(self, tfhub_url, transform_pkl_path,
               organism='human', num_top_features=500):
    self._model = EnformerScoreVariantsRaw(tfhub_url, organism)
    with tf.io.gfile.GFile(transform_pkl_path, 'rb') as f:
      self._transform = joblib.load(f)
    self._num_top_features = num_top_features
    
  def predict_on_batch(self, inputs):
    scores = self._model.predict_on_batch(inputs)
    return self._transform.transform(scores)[:, :self._num_top_features]


# In[33]:


class FastaStringExtractor:
    
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()


def variant_generator(vcf_file, gzipped=False):
  """Yields a kipoiseq.dataclasses.Variant for each row in VCF file."""
  def _open(file):
    return gzip.open(vcf_file, 'rt') if gzipped else open(vcf_file)
    
  with _open(vcf_file) as f:
    for line in f:
      if line.startswith('#'):
        continue
      chrom, pos, id, ref, alt_list = line.split('\t')[:5]
      # Split ALT alleles and return individual variants as output.
      for alt in alt_list.split(','):
        yield kipoiseq.dataclasses.Variant(chrom=chrom, pos=pos,
                                           ref=ref, alt=alt, id=id)


def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)


def variant_centered_sequences(vcf_file, sequence_length, gzipped=False,
                               chr_prefix=''):
  seq_extractor = kipoiseq.extractors.VariantSeqExtractor(
    reference_sequence=FastaStringExtractor(fasta_file))

  for variant in variant_generator(vcf_file, gzipped=gzipped):
    interval = Interval(chr_prefix + variant.chrom,
                        variant.pos, variant.pos)
    interval = interval.resize(sequence_length)
    center = interval.center() - interval.start

    reference = seq_extractor.extract(interval, [], anchor=center)
    alternate = seq_extractor.extract(interval, [variant], anchor=center)

    yield {'inputs': {'ref': one_hot_encode(reference),
                      'alt': one_hot_encode(alternate)},
           'metadata': {'chrom': chr_prefix + variant.chrom,
                        'pos': variant.pos,
                        'id': variant.id,
                        'ref': variant.ref,
                        'alt': variant.alt}}


# In[34]:


def plot_tracks(tracks, interval, height=1.5):
  fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
  for ax, (title, y) in zip(axes, tracks.items()):
    ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
    ax.set_title(title)
    sns.despine(top=True, right=True, bottom=True)
  ax.set_xlabel(str(interval))
  plt.tight_layout()


# ## make predictions for genetic sequences

# In[35]:


target_interval = kipoiseq.Interval('chr11', 35_082_742, 35_197_430)  # @param
target_interval


# In[36]:


model = Enformer(model_path)
fasta_extractor = FastaStringExtractor(fasta_file)
sequence = fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH))


# In[56]:


sequence_one_hot = one_hot_encode(sequence)
predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]


# ## mutate one hot encoding

# In[38]:


print(sequence_one_hot[:5000, :])


# In[39]:


#assert array as a factor of 4
assert sequence_one_hot.shape[1] == 4

#create list to store iterations
iterated_results = []

for start in range(0, SEQUENCE_LENGTH, 1000):
    end = min(start + 1000, SEQUENCE_LENGTH)
    
    iteration_copy = np.copy(sequence_one_hot)
    iteration_copy[start:end, :] = 0.25
    iterated_results.append(iteration_copy)


# In[60]:


# print each iteration or some part of it to verify the changes
for idx, iteration in enumerate(iterated_results):
    print(f"Iteration {idx}:\n", iteration[2000:2005])


# ## calculate delta at TSS

# In[61]:


predictions.shape


# In[13]:


tss_position = np.argmax(predictions[:,4799])
tss_position


# In[29]:


predictions_list = []

for iteration_result in iterated_results:
    prediction = model.predict_on_batch(np.array([iteration_result]))['human'][0]
    predictions_list.append(prediction)

predictions_list


# In[63]:


np.all(predictions[440] == predictions[440,:])


# In[54]:


predictions_list_TSS = []

for iteration_result in iterated_results:
    prediction = model.predict_on_batch(np.array([iteration_result]))['human'][0][440, :]
    predictions_list_TSS.append(prediction)

predictions_list_TSS


# In[67]:


predictions_array_TSS = np.array(predictions_list_TSS)
predictions_array_TSS.shape


# In[86]:


predictions_TSS_CAGE_Keratinocyte_epidermal = predictions_array_TSS[:, 4799]


# In[87]:


plt.plot(predictions_TSS_CAGE_Keratinocyte_epidermal)
plt.show()


# In[74]:


sequence_padding = SEQUENCE_LENGTH // 4
sequence_padding


# In[77]:


end_position_in_sequence_actually_used_for_prediction = SEQUENCE_LENGTH - sequence_padding
end_position_in_sequence_actually_used_for_prediction


# In[78]:


end_position_in_sequence_actually_used_for_prediction - sequence_padding


# In[72]:


SEQUENCE_LENGTH // 2


# In[85]:


values_ = predictions_TSS_CAGE_Keratinocyte_epidermal
plt.plot(values_)
plt.vlines(sequence_padding / 1_000, 0 , np.max(values_), color='red')
plt.vlines(end_position_in_sequence_actually_used_for_prediction / 1_000, 0, np.max(values_), color='red')
plt.show()

