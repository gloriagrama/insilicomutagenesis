#!/usr/bin/env python
# coding: utf-8

# ### imports

# In[6]:


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
import re
import random
import scipy.stats as stats
from scipy.signal import find_peaks 

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[7]:


get_ipython().system('python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices(\'GPU\'))"')


# In[8]:


physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
    print("TensorFlow is using the GPU.")
    print("Available GPU(s):")
    for gpu in physical_devices:
        print(gpu)
else:
    print("TensorFlow is not using the GPU.")


# In[9]:


print(physical_devices)


# In[10]:


transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1'
fasta_file = '/home/jovyan/data4/ggrama_runAI/data/genome.fa'
clinvar_vcf = '/home/jovyan/data4/ggrama_runAI/data/clinvar.vcf.gz'


# ## enformer code

# In[11]:


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


# In[12]:


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


# In[13]:


def plot_tracks(tracks, interval, height=1.5):
  fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
  for ax, (title, y) in zip(axes, tracks.items()):
    ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
    ax.set_title(title)
    sns.despine(top=True, right=True, bottom=True)
  ax.set_xlabel(str(interval))
  plt.tight_layout()


# ## predictions

# In[14]:


#input target seq here
target_interval = kipoiseq.Interval('chr10', 8_054_688, 8_075_198)
target_interval


# In[15]:


model = Enformer(model_path)
fasta_extractor = FastaStringExtractor(fasta_file)
sequence = fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH))
sequence_one_hot = one_hot_encode(sequence)


# In[16]:


#output prediction track for jurkat T cells (track 4831)
predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0][:, 4831]
plt.plot(predictions)
plt.show()


# ## TSS coordinate

# In[17]:


tss_position = np.argmax(predictions)


# In[18]:


(8_075_198 - 8_054_688) / 896


# In[19]:


tss_coordinate = (tss_position * 22.890625) + 8_054_688
tss_coordinate


# ## adjust window to enformer length and center TSS

# In[20]:


tss_position = np.argmax(predictions)
tss_position


# In[21]:


#SEQUENCE_LENGTH//2 +/- tss position 
target_interval = kipoiseq.Interval('chr10', 7_858_447, 8_251_663)
target_interval


# In[22]:


model = Enformer(model_path)
fasta_extractor = FastaStringExtractor(fasta_file)
sequence = fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH))
sequence_one_hot = one_hot_encode(sequence)


# In[23]:


gpu_available = tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
print(f"GPU available: {gpu_available}")


# In[24]:


plt.figure(figsize=(13,7))
baseline_predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0][:, 4831]
plt.plot(baseline_predictions, color='red')
plt.title("Baseline GATA3 Enformer CAGE-seq Predictions", fontweight='bold')
plt.ylabel("CAGE-seq (normalized bin-level expression)")
plt.xlabel("Sequence Index")
plt.show()


# In[25]:


tss_position = np.argmax(baseline_predictions)
tss_position


# ## tile GATA3 and make predictions for iterations (experiment 4831)

# In[26]:


#assert array as a factor of 4
assert sequence_one_hot.shape[1] == 4

#create list to store iterations
iterated_results = []

for start in range(0, SEQUENCE_LENGTH, 128):
    end = min(start + 128, SEQUENCE_LENGTH)
    
    iteration_copy = np.copy(sequence_one_hot)
    iteration_copy[start:end, :] = 0.25
    iterated_results.append(iteration_copy)


# In[27]:


# print each iteration or some part of it to verify the changes
for idx, iteration in enumerate(iterated_results):
    print(f"Iteration {idx}:\n", iteration[128:133, ])


# ### prediction file save

# In[ ]:


predictions_list = []

for i, iteration_result in enumerate(iterated_results):
    prediction = model.predict_on_batch(np.array([iteration_result]))['human'][0][445, :]
    predictions_list.append(prediction)
    
    print(f"Processed sequence {i} / {len(iterated_results)}")

predictions_list


# In[ ]:


predictions_array = np.array(predictions_list)
jurkat_predictions = np.copy(predictions_array[:, 4831])


# In[ ]:


file_path = '/home/jovyan/data4/ggrama_runAI/data/jurkat_predictions.npy'
np.save(file_path, jurkat_predictions)


# ### load prediction file

# In[28]:


# Load the .npy file
jurkat_predictions = np.load('/home/jovyan/data4/ggrama_runAI/data/jurkat_predictions.npy')

# Print the array
print(jurkat_predictions)


# In[29]:


len(jurkat_predictions)


# ## calculate delta at TSS

# In[30]:


baseline_CAGE_TSS = np.max(baseline_predictions)
baseline_CAGE_TSS


# In[31]:


deltas = jurkat_predictions - baseline_CAGE_TSS
deltas


# In[32]:


plt.plot(deltas)


# ## crop out sequence padding

# In[33]:


sequence_padding = SEQUENCE_LENGTH // 4
end_position = SEQUENCE_LENGTH - sequence_padding

plt.plot(deltas)
plt.vlines(sequence_padding / 128, -30 , np.max(deltas), color='red')
plt.vlines(end_position / 128, -30, np.max(deltas), color='red')
plt.show()


# In[34]:


print(sequence_padding) 
print(end_position) 


# In[35]:


print(98304 // 128)
print(294912 // 128)


# In[36]:


adjusted_deltas = deltas[768:2304]


# In[37]:


plt.figure(figsize=(15,7))
plt.plot(adjusted_deltas, color='black')
plt.title("Tiled Deletion Iterations and Enformer CAGE-seq Predictions", fontweight='bold')
plt.ylabel("Delta CAGE-seq Expression from Baseline")
plt.xlabel("Sequence Index")
plt.show()


# ## scaling (sanity check)

# In[38]:


adjusted_seq_start = 7_858_447 + sequence_padding
adjusted_seq_start


# In[39]:


adjusted_seq_end = 8_251_663 - sequence_padding
adjusted_seq_end


# In[40]:


(8153359 - 7956751) / 1536


# In[41]:


tss_coordinate = adjusted_seq_start + ((np.argmin(adjusted_deltas)) * 128)
tss_coordinate


# In[42]:


np.argmin(adjusted_deltas)


# ## GATA3 functional sequences

# pulled from supplementary table 7 in this paper: https://www.cell.com/ajhg/fulltext/S0002-9297(23)00092-7?_returnURL=https%3A%2F%2Flinkinghub.elsevier.com%2Fretrieve%2Fpii%2FS0002929723000927%3Fshowall%3Dtrue

# In[43]:


regions = pd.read_csv('/home/jovyan/data4/ggrama/gata3_fs.txt', 
                      header=None, 
                      sep = '\t')
columns_to_drop = [0, 3, 4] 
regions = regions.drop(columns_to_drop, axis=1)


# In[44]:


regions.rename(columns={1: 'start', 2: 'end'}, inplace=True)
print(regions)


# In[45]:


adjusted_seq_start = int(adjusted_seq_start)
adjusted_seq_end = int(adjusted_seq_end)

conditions = (regions['start'] < adjusted_seq_start) | (regions['end'] > adjusted_seq_end)
regions = regions[~conditions]


# In[46]:


#limitation here, only looking at regions contained within this 200kb window, 17 regions could not be included


# In[47]:


adjusted_regions = (regions - adjusted_seq_start) // 128
adjusted_regions


# In[50]:


relics_array = np.array(adjusted_regions)
flattened_relics = relics_array.ravel().tolist()
unique_relics = list(set(flattened_relics))
unique_relics = sorted(unique_relics)

print(unique_relics)


# In[51]:


def array_to_pairs(relics_array):
    # Reshape the 2x97 array into a flat list
    flat_list = relics_array.flatten()

    pairs_list = []
    
    # Ensure the length of the flat list is even
    if len(flat_list) % 2 != 0:
        return "Error: Length of the array should be even."
    
    for i in range(0, len(flat_list), 2):
        pair = (flat_list[i], flat_list[i + 1])
        pairs_list.append(pair)
    
    return pairs_list



# Convert array to pairs
pairs = array_to_pairs(relics_array)

print(pairs)


# In[52]:


plt.figure(figsize=(13,7))

plt.plot(adjusted_deltas, color='black')
regions = pairs
y_min = 6
y_values = np.array([y_min]*3)*np.array([1, 1.25, 1.5])
prev_y = y_min
colors = ['red']
color_index = 0

for start, end in regions[:1]:
    # Use the current color from the list
    current_color = colors[color_index]
    
    # Draw the horizontal line with the current color
    plt.hlines(prev_y, start, end, colors=current_color, linewidth=6, label='Functional Region')
    
    # Choose next y as one of the other available y-values
    remain_y = list(y_values)
    remain_y.pop(list(y_values).index(prev_y))
    prev_y = np.random.choice(remain_y)
    
    # Update the color index (cycling through the list)
    color_index

for start, end in regions[1:]:
    # Use the current color from the list
    current_color = colors[color_index]
    
    # Draw the horizontal line with the current color
    plt.hlines(prev_y, start, end, colors=current_color, linewidth=6)
    
    # Choose next y as one of the other available y-values
    remain_y = list(y_values)
    remain_y.pop(list(y_values).index(prev_y))
    prev_y = np.random.choice(remain_y)
    
    # Update the color index (cycling through the list)
    color_index

plt.title('Known GATA3 Functional Sequences Plotted Against Tiled Deletions', fontweight='bold')
plt.xlabel('Sequence Position')
plt.ylabel('Delta CAGE_seq Expression at TSS')
plt.legend(loc='lower right')
plt.show()


# In[53]:


start = []
for index in adjusted_regions:
    index = adjusted_regions['start']
    start.append(index)

print(start)


# In[54]:


end = []
for index in adjusted_regions:
    index = adjusted_regions['end']
    end.append(index)

print(end)


# ## does the magnitude of the predicted expression differences look higher in the known regulatory regions versus the known non-regulatory regions that are important for GATA3 gene expression?

# In[55]:


#sums values across start and end regions, absolute values 

start_indices = adjusted_regions['start']
end_indices = adjusted_regions['end']

reg_mags = []  

for start, end in zip(start_indices, end_indices):
    if start == end:
        reg_mags.append(adjusted_deltas[start])
    else:
        reg_mag = abs(sum(adjusted_deltas[start:end]) / (end - start))  
        reg_mags.append(reg_mag)

print(reg_mags)


# ## background distribution

# In[56]:


data = adjusted_deltas
regions = [(adjusted_regions['start'].iloc[i], adjusted_regions['end'].iloc[i]) for i in range(len(adjusted_regions))]

num_datasets = 100

# Function to generate randomized regions with consistent length but different starting positions
def generate_shifted_randomized_regions(data, regions):
    randomized_regions = []
    
    for start, end in regions:
        region_length = end - start
        
        # List comprehension to generate the shifted starting positions
        random_starts = [np.random.randint(0, len(data) - region_length) for _ in range(num_datasets)]
        
        # Create the shifted regions
        shifted_regions = [(random_start, random_start + region_length) for random_start in random_starts]
        
        randomized_regions.append(shifted_regions)
    
    return randomized_regions

# Generate randomized regions with consistent region lengths but different starting positions
shifted_randomized_regions = generate_shifted_randomized_regions(adjusted_deltas, regions)


# In[57]:


shifted_randomized_regions


# In[58]:


# Plot the first 3 iterations of randomized regions against the original data and known GATA3 functional sequences
for i in range(3):
    randomized_regions = shifted_randomized_regions[i]  # Access randomized regions for the current iteration

    plt.plot(adjusted_deltas, color='blue')

    # Plot known GATA3 functional sequences
    for start, end in regions:
        plt.vlines(x=[start, end], ymin=np.min(adjusted_deltas), ymax=np.max(adjusted_deltas), color='red', linewidth=1.2, label='Known Sequence')
        break
        

    # Plot randomized regions for the current iteration in green
    for start, end in randomized_regions:
        plt.vlines(x=[start, end], ymin=np.min(adjusted_deltas), ymax=np.max(adjusted_deltas), color='green', linewidth=0.25, label='Randomized Regions')


    plt.title(f'Functional Sequence {i+1} with Randomized Regions')
    plt.xlabel('Sequence Position')
    plt.ylabel('Delta CAGE_seq Expression at TSS')
    plt.show()


# In[59]:


mean_across_iterations = []

for region_idx in range(len(regions)):
    region_means = []

    for i in range(num_datasets):
        if i < len(shifted_randomized_regions):
            randomized_regions = shifted_randomized_regions[i]
            if region_idx < len(randomized_regions):
                start, end = randomized_regions[region_idx]
                region_values = adjusted_deltas[start:end]
                
                # Check if region_values is not empty before calculating mean
                if len(region_values) > 0:
                    region_mean = np.mean(region_values)
                    region_means.append(region_mean)
    
    if region_means:  # Check if region_means list is not empty
        mean_across_iterations.append(np.mean(region_means))
    else:
        mean_across_iterations.append(0)  # Placeholder value if no data available for the region

# The list mean_across_iterations now contains the mean for each region across all iterations

mean_across_iterations = [abs(value) for value in mean_across_iterations]


# In[60]:


from scipy.stats import ttest_rel

expected_means = mean_across_iterations
observed_means = reg_mags

t_stat, p_val = ttest_rel(expected_means, observed_means)

if p_val < 0.05:
    print("The observed region means are statistically significant.")
else:
    print('The observed region means are not statistically significant.')


# In[61]:


plt.figure(figsize=(13,7))

def run_baseline_ttest(expected_means, observed_means, TEST_NAME, out_plots, log_file, alloc=''):
 
    # Create the violin plot
    df = pd.DataFrame({"expected": expected_means, "observed": observed_means})
    sns.violinplot(data=df, palette=('white', 'salmon'))
    
    # Connect paired data points with lines
    for i in np.random.randint(0, len(expected_means), 100):
        plt.plot([0, 1], [expected_means[i], observed_means[i]], 'k-', alpha=0.3)
    
    plt.title('Means of Background Distribution vs. Observed Means of Known Functional Regions', fontweight='bold')
    plt.ylabel('Delta CAGE-seq Expression at TSS')
    plt.show()

    # Paired t-test calculation
    t_stat, p_val = stats.ttest_rel(observed_means, expected_means)

    # Print results and save to log file
    print("\nMean expected MSE:", np.mean(expected_means), "Mean observed MSE:", np.mean(observed_means),
          file=log_file, flush=True)
    print(f"T-statistic: {t_stat}", file=log_file, flush=True)
    print(f"P-value: {p_val}\n", file=log_file, flush=True)
    print(f"Saved ", out_plots + f'{TEST_NAME}{alloc}_obsVSexp_violin.png\n', file=log_file, flush=True)
run_baseline_ttest(expected_means, observed_means, "Test", "output_plots/", log_file=open("log.txt", "w"))


# ## characterizing uncalled peaks through quantifying uncertainty in models predictions by creating a jittered data set

# In[62]:


# had to split up jitters because it was too much to process at once
jitter_128 = kipoiseq.Interval('chr10', 7_858_447 + 128, 8_251_663 + 128)
jitter_256 = kipoiseq.Interval('chr10', 7_858_447 + 256, 8_251_663 + 256)
jitter_128_neg = kipoiseq.Interval('chr10', 7_858_447 - 128, 8_251_663 - 128)
jitter_256_neg = kipoiseq.Interval('chr10', 7_858_447 - 256, 8_251_663 - 256)


# In[63]:


j128_seq = fasta_extractor.extract(jitter_128.resize(SEQUENCE_LENGTH))
j256_seq = fasta_extractor.extract(jitter_256.resize(SEQUENCE_LENGTH))
j128_neg_seq = fasta_extractor.extract(jitter_128_neg.resize(SEQUENCE_LENGTH))
j256_neg_seq = fasta_extractor.extract(jitter_256_neg.resize(SEQUENCE_LENGTH))


# In[64]:


one_hot_j128 = one_hot_encode(j128_seq)
one_hot_j256 = one_hot_encode(j256_seq)
one_hot_j128_neg = one_hot_encode(j128_neg_seq)
one_hot_j256_neg = one_hot_encode(j256_neg_seq)


# In[65]:


# iterations for each jitter

#assert array as a factor of 4
assert one_hot_j128.shape[1] == 4

#create list to store iterations
j128_iteration = []

for start in range(0, SEQUENCE_LENGTH, 128):
    end = min(start + 128, SEQUENCE_LENGTH)
    
    iteration_copy = np.copy(one_hot_j128)
    iteration_copy[start:end, :] = 0.25
    j128_iteration.append(iteration_copy)
    

#assert array as a factor of 4
assert one_hot_j256.shape[1] == 4

#create list to store iterations
j256_iteration = []

for start in range(0, SEQUENCE_LENGTH, 128):
    end = min(start + 128, SEQUENCE_LENGTH)
    
    iteration_copy = np.copy(one_hot_j256)
    iteration_copy[start:end, :] = 0.25
    j256_iteration.append(iteration_copy)
    
    
#assert array as a factor of 4
assert one_hot_j128_neg.shape[1] == 4

#create list to store iterations
j128_neg_iteration = []

for start in range(0, SEQUENCE_LENGTH, 128):
    end = min(start + 128, SEQUENCE_LENGTH)
    
    iteration_copy = np.copy(one_hot_j128_neg)
    iteration_copy[start:end, :] = 0.25
    j128_neg_iteration.append(iteration_copy)
    
    
#assert array as a factor of 4
assert one_hot_j256_neg.shape[1] == 4

#create list to store iterations
j256_neg_iteration = []

for start in range(0, SEQUENCE_LENGTH, 128):
    end = min(start + 128, SEQUENCE_LENGTH)
    
    iteration_copy = np.copy(one_hot_j256_neg)
    iteration_copy[start:end, :] = 0.25
    j256_neg_iteration.append(iteration_copy)


# In[66]:


# print each iteration or some part of it to verify the changes
for idx, iteration in enumerate(j256_neg_iteration):
    print(f"Iteration {idx}:\n", iteration[0:128, ])


# ### jitter prediction file save

# In[ ]:


predictions_j128 = []

for i, iteration_result in enumerate(j128_iteration):
    prediction = model.predict_on_batch(np.array([iteration_result]))['human'][0][444, :]
    predictions_j128.append(prediction)
    
    print(f"Processed sequence {i} / {len(iterated_results)}")

predictions_j128


# In[ ]:


predictions_j256 = []

for i, iteration_result in enumerate(j256_iteration):
    prediction = model.predict_on_batch(np.array([iteration_result]))['human'][0][443, :]
    predictions_j256.append(prediction)
    
    print(f"Processed sequence {i} / {len(iterated_results)}")

predictions_j256


# In[ ]:


predictions_j128_neg = []

for i, iteration_result in enumerate(j128_neg_iteration):
    
    prediction = model.predict_on_batch(np.array([iteration_result]))['human'][0][446, :]
    predictions_j128_neg.append(prediction)
    
    print(f"Processed sequence {i} / {len(iterated_results)}")

predictions_j128_neg


# In[ ]:


predictions_j256_neg = []

for i, iteration_result in enumerate(j256_neg_iteration):
    prediction = model.predict_on_batch(np.array([iteration_result]))['human'][0][447, :]
    predictions_j256_neg.append(prediction)
    
    print(f"Processed sequence {i} / {len(iterated_results)}")

predictions_j256_neg


# In[ ]:


j128 = np.array(predictions_j128)
j128 = np.copy(j128[:, 4831])

j256 = np.array(predictions_j256)
j256 = np.copy(j256[:, 4831])

j128_neg = np.array(predictions_j128_neg)
j128_neg = np.copy(j128_neg[:, 4831])

j256_neg = np.array(predictions_j256_neg)
j256_neg = np.copy(j256_neg[:, 4831])


# In[ ]:


file_path = '/home/jovyan/data4/ggrama_runAI/data/j128.npy'
np.save(file_path, j128)


# In[ ]:


file_path = '/home/jovyan/data4/ggrama_runAI/data/j256.npy'
np.save(file_path, j256)


# In[ ]:


file_path = '/home/jovyan/data4/ggrama_runAI/data/j128_neg.npy'
np.save(file_path, j128_neg)


# In[ ]:


file_path = '/home/jovyan/data4/ggrama_runAI/data/j256_neg.npy'
np.save(file_path, j256_neg)


# ## jitter file load

# In[82]:


# Load the .npy file
j128_saved = np.load('/home/jovyan/data4/ggrama_runAI/data/j128.npy')

# Print the array
print(j128_saved)


# In[83]:


# Load the .npy file
j256_saved = np.load('/home/jovyan/data4/ggrama_runAI/data/j256.npy')

# Print the array
print(j256_saved)


# In[84]:


# Load the .npy file
j128_neg_saved = np.load('/home/jovyan/data4/ggrama_runAI/data/j128_neg.npy')

# Print the array
print(j128_neg_saved)


# In[85]:


# Load the .npy file
j256_neg_saved = np.load('/home/jovyan/data4/ggrama_runAI/data/j256_neg.npy')

# Print the array
print(j256_neg_saved)


# In[86]:


trimmed_predictions_j128 = np.copy(j128_saved[769:2301])
trimmed_predictions_j256 = np.copy(j256_saved[768:2300])
trimmed_predictions_j128_neg = np.copy(j128_neg_saved[771:2303])
trimmed_predictions_j256_neg = np.copy(j256_neg_saved[772:2304])


# In[87]:


print(len(trimmed_predictions_j128))
print(len(trimmed_predictions_j256))
print(len(trimmed_predictions_j128_neg))
print(len(trimmed_predictions_j256_neg))


# In[90]:


x = np.arange(len(trimmed_predictions_j128))

plt.plot(x, trimmed_predictions_j128, label='+128 bp', alpha=0.7)
plt.plot(x, trimmed_predictions_j256, label='+256 bp', alpha=0.7)
plt.plot(x, trimmed_predictions_j128_neg, label='−128 bp', alpha=0.7)
plt.plot(x, trimmed_predictions_j256_neg, label='−256 bp', alpha=0.7)


plt.xlabel('Sequence Index')
plt.ylabel('Delta CAGE-seq Expression at TSS')
plt.title('Overlay of Jittered Arrays')

plt.legend()

plt.show()


# In[91]:


trimmed_jitters = [trimmed_predictions_j128, trimmed_predictions_j256, trimmed_predictions_j128_neg, trimmed_predictions_j256_neg]


# In[92]:


jitter_deltas = []

for array in trimmed_jitters:
    delta = array - baseline_CAGE_TSS
    jitter_deltas.append(delta)

jitter_deltas = np.array(jitter_deltas)
print(jitter_deltas)


# In[93]:


x = np.arange(len(trimmed_predictions_j128))

plt.figure(figsize=(10, 6))

for i, array in enumerate(jitter_deltas):
    plt.plot(x, array, label=f'Array {i + 1}', alpha=0.7)

plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.title('Jittered Tiled Deletions')
plt.xlabel('Sequence Position')
plt.ylabel('Delta CAGE_seq Expression at TSS')

plt.show()


# ## standard error of mean

# In[94]:


df = pd.DataFrame(jitter_deltas)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df)


# In[95]:


#calculate sem across arrays
jitter_2d = np.vstack([trimmed_predictions_j128, trimmed_predictions_j256, trimmed_predictions_j128_neg, trimmed_predictions_j256_neg])
print(jitter_2d)

# ddof = 1 since data is subset of population
std_devs = np.std(jitter_2d, axis=0, ddof=1)

# Number of arrays (samples)
n = 4

# Calculate the standard error of the mean at each position
sem = std_devs / np.sqrt(n)

# Print the SEM values
print("Standard Error of the Mean at each position:", sem)


# In[96]:


min_value = np.min(jitter_deltas)
min_indices = np.unravel_index(np.where(jitter_deltas == min_value), jitter_deltas.shape)
print(min_indices)


# In[97]:


# Parameters
confidence = 0.95
n = jitter_deltas.shape[0]
dof = n - 1

# Mean, SEM, and t critical value calculation
means = np.mean(jitter_deltas, axis=0)
stderr = stats.sem(jitter_deltas, axis=0)
t_critical = stats.t.ppf((1 + confidence) / 2., dof)

# Confidence Intervals
margin_of_error = t_critical * stderr
ci_lower = means - margin_of_error
ci_upper = means + margin_of_error

# Results
print("Means for each column:", means)
print("Confidence Intervals for each column:", (ci_lower, ci_upper))


# In[98]:


plt.figure(figsize=(13, 7))
plt.plot(x, means, label='Mean', color='b')
plt.fill_between(x, ci_lower, ci_upper, color='b', alpha=0.2, label='95% CI')
labels = ('+128 bp', '+256 bp', '−128 bp', '−256 bp')

for i, array in enumerate(jitter_deltas):
    plt.plot(x, array, label=labels[i], alpha=0.7)

plt.xlabel('Sequence Index')
plt.ylabel('Delta CAGE-seq Expression at TSS')
plt.title('Means with 95% Confidence Intervals per Bin', fontweight='bold')
plt.legend()
plt.xlim([0, 200])
plt.ylim([-5, 5])
plt.grid(True)
plt.show()


# In[99]:


t_stat = means / stderr

# Calculate the p-values for each column
p_values = stats.t.sf(np.abs(t_stat), dof) * 2  # Two-tailed p-value

# Results
print("T-statistics for each column:", t_stat)
print("P-values for each column:", p_values)


# In[101]:


import statsmodels
from statsmodels.stats.multitest import multipletests
FDR = statsmodels.stats.multitest.multipletests(p_values, 
                                          alpha=0.3, 
                                          method='fdr_bh', 
                                          maxiter=1, 
                                          is_sorted=False, 
                                          returnsorted=False)


# In[102]:


q_values = FDR[1]
q_values


# In[103]:


significant_q = np.where(q_values <= 0.3)[0]

# Output the significant column indexes
print(significant_q)


# In[104]:


significant_p = np.where(p_values <= 0.05)[0]

# Output the significant column indexes
print(significant_p)


# In[105]:


plt.figure(figsize=(13, 7))

# Plot the main data
plt.plot(adjusted_deltas, color='black')

x_coordinates = significant_q
y_min = 6
y_values = np.array([y_min]*3)*np.array([1, 1.25, 1.5])
prev_y = y_min
colors = ['blue']  

# Plot the main data
plt.plot(adjusted_deltas, color='black')

for x in x_coordinates:
    # Choose a new y-coordinate from the available y-values
    if prev_y is not None:  # If there was a previous y-coordinate
        remain_y = list(y_values)  # Create a list of remaining y-values
        remain_y.remove(prev_y)  # Remove the last used y-value from the list

        # Randomly select a new y-value from the remaining options
        current_y = np.random.choice(remain_y)
    else:
        # If no previous y-coordinate exists (first iteration):
        current_y = np.random.choice(y_values)

    plt.plot(x, current_y, marker='o', color='blue', markersize=5, alpha = 0.5, label=' Computationally Significant Region' if x == x_coordinates[0] else "")
    
    # Update prev_y for the next iteration
    prev_y = current_y

# Bold the title and labels
plt.title('Computationally Significant Peak Regions Accounting for FDR', fontweight='bold')
plt.xlabel('Sequence Position')
plt.ylabel('Delta CAGE-seq Expression at TSS')

# Bold the legend labels
plt.legend(loc = 'lower right')

plt.show()


# In[106]:


plt.plot(adjusted_deltas)
plt.vlines(significant_q, 
           np.min(adjusted_deltas), 
           np.max(adjusted_deltas), 
           color='red',
           linewidth=0.25, 
           alpha=0.5)
plt.title('Computationally Significant Peak Regions Adjusted for False Discovery Rate (somewhat)')
plt.xlabel('Sequence Position')
plt.ylabel('Delta CAGE_seq Expression at TSS')
plt.show()


# In[107]:


significant_columns = np.where(p_values <= 0.05)[0]

# Output the significant column indexes
print(significant_columns)


# In[108]:


plt.plot(adjusted_deltas)
plt.vlines(significant_columns, 
           np.min(adjusted_deltas), 
           np.max(adjusted_deltas), 
           color='red',
           linewidth=0.25, 
           alpha=0.5)
plt.title('Computationally "Significant" Peak Regions')
plt.xlabel('Sequence Position')
plt.ylabel('Delta CAGE_seq Expression at TSS')
plt.show()


# In[109]:


plt.plot(adjusted_deltas)
plt.vlines(significant_q, 
           np.min(adjusted_deltas), 
           np.max(adjusted_deltas), 
           color='red',
           linewidth=0.5, 
           alpha=0.5,
           label = 'Computationally Determined Regions')
plt.vlines(adjusted_regions,
           np.min(deltas), 
           np.max(deltas), 
           color='purple',
           linewidth=0.5, 
           alpha=0.5, 
           label ='Ground Truth Regions')
plt.title('Known Functional & Computationally Significant Peak Regions')
plt.xlabel('Sequence Index')
plt.ylabel('Delta CAGE_seq Expression at TSS')
plt.legend(loc ='lower left')
plt.show()


# In[110]:


common_elements = [item for item in significant_q if item in relics_array]

print(f"Elements in computed regions that are also in ground truth regions: {common_elements}")


# In[111]:


adjusted_regions


# In[112]:


relics_array


# In[113]:


pairs


# In[114]:


# Function to fill in integers between start and end values in each range
def fill_in_range_regions(region_list):
    filled_numbers = []

    for start, end in region_list:
        filled_numbers.extend(range(start, end + 1))

    return filled_numbers

# List of regions as tuples (start, end)
regions = pairs

# Fill in the integers between ranges in each region
filled_integers = fill_in_range_regions(regions)

print(f"Integers filled within the ranges: {filled_integers}")


# In[115]:


filled_integers = list(set(filled_integers))
sorted(filled_integers)


# In[117]:


relics_fs_coordinates = filled_integers


# In[118]:


relics_non_functional = []

for i in range(1536):
    if i not in relics_fs_coordinates:
        relics_non_functional.append(i)

print(relics_non_functional)


# In[119]:


len(filled_integers)


# ## log odds ratio

# In[120]:


significant_q


# In[121]:


experimental_fs_coordinates = significant_q


# In[122]:


# A
shared_fs = [item for item in experimental_fs_coordinates if item in relics_fs_coordinates]

print(f"Functional regions in computed regions that are also in ground truth regions: {shared_fs}")


# In[123]:


experimental_non_functional = []

for i in range(1536):
    if i not in experimental_fs_coordinates:
        experimental_non_functional.append(i)

print(experimental_non_functional)


# In[124]:


relics_non_functional = []

for i in range(1536):
    if i not in relics_fs_coordinates:
        relics_non_functional.append(i)

print(relics_non_functional)


# In[125]:


# C
relics_exp_non_functional = [item for item in experimental_non_functional if item in relics_fs_coordinates]

print(f"Experimentally non functional regions that are actually ground truth regions: {relics_exp_non_functional}")


# In[126]:


# B
non_relics_exp_functional = [item for item in experimental_fs_coordinates if item in relics_non_functional]

print(f"Experimentally functional regions that are insignificant RELICS regions: {relics_exp_non_functional}")


# In[127]:


# D: Non-functional coordinates in computed regions that are also in ground truth regions
shared_non_fs = [item for item in experimental_non_functional if item in relics_non_functional]
print(f"Non-functional coordinates in computed regions that are also in ground truth regions: {shared_non_fs}")


# In[128]:


print(len(experimental_fs_coordinates))
print(len(experimental_non_functional))
print(len(relics_fs_coordinates))
print(len(relics_non_functional))


# In[129]:


print(len(shared_fs))
print(len(non_relics_exp_functional))
print(len(relics_exp_non_functional))
print(len(shared_non_fs))


# In[130]:


125+407


# In[131]:


56+948


# In[132]:


A = len(shared_fs) / 532
B = len(non_relics_exp_functional) / 532
C = len(relics_exp_non_functional) / 1004
D = len(shared_non_fs) / 1004


# In[133]:


table = [A, B], [C, D]
table


# In[134]:


from scipy.stats import fisher_exact

fisher_exact(table, alternative='two-sided')


# In[135]:


df = pd.DataFrame(table)
df.columns = ['CRISPR functional', 'CRISPR non functional']
df.index = ['in silico functional', 'in silico non functional']
df


# In[136]:


# Create the heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(df, annot=True, cmap='Blues', fmt=".2f", linewidths=.5,
                 cbar_kws={'label': 'Enrichment'}, annot_kws={"size": 16})

plt.title('in silico vs. CRISPR Tiling Contingency Table at Optimized Alpha', fontweight='bold')

labels = [['True Positive', 'FDR'],
          ['False Negative', 'True Negative']]

# Label colors
colors = [['black', 'white'],
          ['black', 'white']]

# Add the labels above the numbers with specified colors
for i in range(len(labels)):
    for j in range(len(labels[i])):
        ax.text(j + 0.5, i + 0.15, labels[i][j], ha='center', va='center', 
                fontsize=12, color=colors[i][j])
        
# Show the plot
plt.show()


# In[137]:


from scipy.stats import fisher_exact

fisher_exact(table, alternative='two-sided')


# ## genome cut proportion

# In[138]:


type(experimental_fs_coordinates)


# In[139]:


total_functional_coordindates = []

total_functional_coordindates = list(experimental_fs_coordinates) + list(relics_fs_coordinates)


# In[140]:


total_functional_coordindates = list(set(total_functional_coordindates))
total_functional_coordindates.sort()
print(total_functional_coordindates)


# In[141]:


from collections import Counter

def has_duplicates(total_functional_coordindates):
    return any(count > 1 for count in Counter(total_functional_coordindates).values())

print(has_duplicates(total_functional_coordindates)) 


# In[142]:


proportion_kept = len(total_functional_coordindates) / 1536 


# In[143]:


proportion_kept


# In[144]:


SEQUENCE_LENGTH / 2


# In[145]:


196608 * proportion_kept


# ## determining optimized alpha

# In[146]:


experimental_fs_coordinates = significant_columns


# In[147]:


significant_columns


# In[148]:


relics_fs_coordinates = filled_integers


# In[149]:


# A
shared_fs = [item for item in experimental_fs_coordinates if item in relics_fs_coordinates]

print(f"Functional regions in computed regions that are also in ground truth regions: {shared_fs}")


# In[150]:


experimental_non_functional = []

for i in range(1536):
    if i not in experimental_fs_coordinates:
        experimental_non_functional.append(i)

print(experimental_non_functional)


# In[151]:


relics_non_functional = []

for i in range(1536):
    if i not in relics_fs_coordinates:
        relics_non_functional.append(i)

print(relics_non_functional)


# In[152]:


# D
shared_non_fs = [item for item in experimental_non_functional if item in relics_non_functional]

print(f"Non functional coordinates in computed regions that are also in ground truth regions: {shared_non_fs}")


# In[153]:


# C
# good that this is high (kinda the point of doing this??? not sure if this is a good measure)
relics_exp_non_functional = [item for item in experimental_non_functional if item in relics_fs_coordinates]

print(f"Experimentally non functional regions that are actually ground truth regions: {relics_exp_non_functional}")


# In[154]:


# B
non_relics_exp_functional = [item for item in experimental_fs_coordinates if item in relics_non_functional]

print(f"Experimentally functional regions that are insignificant RELICS regions: {relics_exp_non_functional}")


# In[155]:


print(len(experimental_fs_coordinates))
print(len(experimental_non_functional))
print(len(relics_fs_coordinates))
print(len(relics_non_functional))


# In[156]:


print(len(shared_fs))
print(len(non_relics_exp_functional))
print(len(relics_exp_non_functional))
print(len(shared_non_fs))


# In[157]:


110+71


# In[158]:


283+1072


# In[166]:


A = len(shared_fs) / 181
B = len(non_relics_exp_functional) / 1355
C = len(relics_exp_non_functional) / 181
D = len(shared_non_fs) / 1355

prop_table = [A, B], [C, D]


# In[167]:


A = len(shared_fs)
B = len(non_relics_exp_functional)
C = len(relics_exp_non_functional)
D = len(shared_non_fs)

table = [A, B], [C, D]
table

from scipy.stats import fisher_exact

fisher_exact(table, alternative='two-sided')


# In[162]:


from scipy.stats import fisher_exact

fisher_exact(table, alternative='two-sided')


# In[169]:


df = pd.DataFrame(prop_table)
df.columns = ['RELICS functional', 'RELICS non functional']
df.index = ['in silico functional', 'in silico non functional']
df


# In[170]:


df = pd.DataFrame(table)
df.columns = ['RELICS functional', 'RELICS non functional']
df.index = ['in silico functional', 'in silico non functional']
df


# In[171]:


# Create the heatmap
plt.figure(figsize=(10, 8))
ax = sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, cbar_kws={'label': 'Enrichment'})  

plt.title('Experimental vs. Control Contingency Table without Adjusting FDR')

# Show the plot
plt.show()


# In[172]:


import statsmodels
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy.stats import fisher_exact

# Define a list of different alpha values
alpha_values = [0.05, 0.1, 0.15, 0.2]


for alpha in alpha_values:
    # Perform the FDR correction
    FDR = multipletests(p_values, 
                        alpha=alpha, 
                        method='fdr_bh', 
                        maxiter=1, 
                        is_sorted=False, 
                        returnsorted=False)
    
    # Extract q-values
    q_values = FDR[1]

    # Find significant q-values with the threshold of the current alpha value
    significant_q = np.where(q_values <= alpha)[0]

    # The significant column indexes are the experimental functional coordinates
    experimental_fs_coordinates = significant_q

    # Output the significant column indexes
    print(f"Alpha: {alpha}")
    print(f"Significant column indexes (q_values <= {alpha}): {significant_q}")

    # A: Functional regions in computed regions that are also in ground truth regions
    shared_fs = [item for item in experimental_fs_coordinates if item in relics_fs_coordinates]
    print(f"Functional regions in computed regions that are also in ground truth regions: {shared_fs}")

    # Experimental non-functional regions
    experimental_non_functional = [i for i in range(1536) if i not in experimental_fs_coordinates]
    print(f"Experimental non-functional regions: {experimental_non_functional}")

    # D: Non-functional coordinates in computed regions that are also in ground truth regions
    shared_non_fs = [item for item in experimental_non_functional if item in relics_non_functional]
    print(f"Non-functional coordinates in computed regions that are also in ground truth regions: {shared_non_fs}")

    # C: Experimentally non-functional regions that are actually ground truth regions
    relics_exp_non_functional = [item for item in experimental_non_functional if item in relics_fs_coordinates]
    print(f"Experimentally non-functional regions that are actually ground truth regions: {relics_exp_non_functional}")

    # B: Experimentally functional regions that are insignificant RELICS regions
    non_relics_exp_functional = [item for item in experimental_fs_coordinates if item in relics_non_functional]
    print(f"Experimentally functional regions that are insignificant RELICS regions: {non_relics_exp_functional}")

    # Calculate A, B, C, D
    A = len(shared_fs)
    B = len(non_relics_exp_functional)
    C = len(relics_exp_non_functional)
    D = len(shared_non_fs)

    # Create contingency table
    table = [[A, B], [C, D]]

    # Perform Fisher's Exact Test
    fisher_result = fisher_exact(table, alternative='two-sided')

    print(f"Contingency Table: {table}")
    print(f"Fisher's Exact Test result for alpha {alpha}: {fisher_result}")

    print("\n")


# In[173]:


import statsmodels
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy.stats import fisher_exact

# Define a list of different alpha values
alpha_values = np.arange(0.05, 0.26, 0.01)

# Define costs for Type I and Type II errors
c1 = 0  # False positive
c2 = 1  # False negative

costs = []

results = []  

for alpha in alpha_values:
    # Perform the FDR correction
    FDR = multipletests(p_values, 
                        alpha=alpha, 
                        method='fdr_bh', 
                        maxiter=1, 
                        is_sorted=False, 
                        returnsorted=False)
    
    # Extract q-values
    q_values = FDR[1]

    # Find significant q-values with the threshold of the current alpha value
    significant_q = np.where(q_values <= alpha)[0]

    # The significant column indexes are the experimental functional coordinates
    experimental_fs_coordinates = significant_q

    # Output the significant column indexes
    print(f"Alpha: {alpha}")

    # A: Functional regions in computed regions that are also in ground truth regions
    shared_fs = [item for item in experimental_fs_coordinates if item in relics_fs_coordinates]

    # Experimental non-functional regions
    experimental_non_functional = [i for i in range(1536) if i not in experimental_fs_coordinates]

    # D: Non-functional coordinates in computed regions that are also in ground truth regions
    shared_non_fs = [item for item in experimental_non_functional if item in relics_non_functional]

    # C: Experimentally non-functional regions that are actually ground truth regions
    relics_exp_non_functional = [item for item in experimental_non_functional if item in relics_fs_coordinates]

    # B: Experimentally functional regions that are insignificant RELICS regions
    non_relics_exp_functional = [item for item in experimental_fs_coordinates if item in relics_non_functional]

    # Calculate A, B, C, D
    A = len(shared_fs)
    B = len(non_relics_exp_functional)
    C = len(relics_exp_non_functional)
    D = len(shared_non_fs)

    # Create contingency table
    table = [[A, B], [C, D]]

    # Perform Fisher's Exact Test
    fisher_result = fisher_exact(table, alternative='two-sided')
    fisher_statistic = fisher_result[0]  
    fisher_p_value = fisher_result[1]  # P-value to assess significance

    print(f"Contingency Table: {table}")
    print(f"Fisher's Exact Test result for alpha {alpha}: {fisher_result}")

    # Calculate Type I and Type II errors
    type_i_errors = B
    type_ii_errors = C

    # Calculate total cost
    total_cost = c1 * type_i_errors + c2 * type_ii_errors
    results.append((alpha, total_cost, fisher_p_value, fisher_statistic))

    print(f"Total cost for alpha {alpha}: {total_cost}")
    print(f"Fisher's Exact Test p-value for alpha {alpha}: {fisher_p_value}\n")

# Find the alpha with the minimum cost and consider Fisher's Exact Test p-value
optimal_result = max(results, key=lambda x: (x[1] == min([r[1] for r in results]), x[3]))

optimal_alpha = optimal_result[0]
min_cost = optimal_result[1]
min_p_value = optimal_result[2]
fisher_statistic_val = optimal_result[3]

print(f"Optimal alpha value: {optimal_alpha} with minimum cost: {min_cost} and Fisher's p-value: {min_p_value}")
print(f"Fisher's Exact Test statistic for optimal alpha: {fisher_statistic_val}")


# In[175]:


import statsmodels
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt

# Generate alpha values from 0.05 to 0.25 in steps of 0.01
alpha_values = np.arange(0.05, 0.26, 0.01)

results = []  # Initialize results as a list
fisher_statistics = []  # List to store Fisher statistics for plotting
fisher_results = []  # List to store Fisher result (statistic, p-value)

for alpha in alpha_values:
    # Perform the FDR correction
    FDR = multipletests(p_values,
                        alpha=alpha,
                        method='fdr_bh',
                        maxiter=1,
                        is_sorted=False,
                        returnsorted=False)

    # Extract q-values
    q_values = FDR[1]

    # Find significant q-values with the threshold of the current alpha value
    significant_q = np.where(q_values <= alpha)[0]

    # The significant column indexes are the experimental functional coordinates
    experimental_fs_coordinates = significant_q

    # A: Functional regions intersecting with ground truth
    shared_fs = [item for item in experimental_fs_coordinates if item in relics_fs_coordinates]

    # Experimental non-functional regions
    experimental_non_functional = [i for i in range(1536) if i not in experimental_fs_coordinates]

    # D: Non-functional coordinates in computed regions that are also in ground truth regions
    shared_non_fs = [item for item in experimental_non_functional if item in relics_non_functional]

    # C: Experimentally non-functional regions that are actually ground truth regions
    relics_exp_non_functional = [item for item in experimental_non_functional if item in relics_fs_coordinates]

    # B: Experimentally functional regions that are insignificant RELICS regions
    non_relics_exp_functional = [item for item in experimental_fs_coordinates if item in relics_non_functional]

    # Calculate A, B, C, D
    A = len(shared_fs)
    B = len(non_relics_exp_functional)
    C = len(relics_exp_non_functional)
    D = len(shared_non_fs)

    left = A + C
    right = B + D

    # Calculate A, B, C, D
    A = len(shared_fs)
    B = len(non_relics_exp_functional)
    C = len(relics_exp_non_functional)
    D = len(shared_non_fs)

    # Create contingency table
    table = [[A, B], [C, D]]

    # Perform Fisher's Exact Test
    fisher_result = fisher_exact(table, alternative='two-sided')
    fisher_statistic = fisher_result[0]  # Odds ratio
    fisher_p_value = fisher_result[1]    # P-value to assess significance


    fisher_results.append((fisher_statistic, fisher_p_value))

    # Append Fisher statistic for plotting
    fisher_statistics.append(fisher_statistic if not np.isnan(fisher_statistic) else 0)

    # Append to results for further analysis
    results.append((alpha, fisher_p_value, fisher_statistic))  # Append result

# Convert fisher_statistics to NumPy array for easy manipulation (if needed)
fisher_statistics = np.array(fisher_statistics)

# Plot the Fisher statistics against alpha values
plt.figure(figsize=(10, 5))
plt.plot(alpha_values, fisher_statistics, marker='o', linestyle='-', color='b')
plt.title("Fisher's Exact Test Statistic vs Alpha Values")
plt.xlabel("Alpha Values")
plt.ylabel("Fisher's Exact Test Statistic")
plt.grid()
plt.xticks(alpha_values)  # Set x-ticks to be the alpha values for clarity
plt.show()

optimal_result = max(results, key=lambda x: (x[1] == min([r[1] for r in results]), x[2]))

optimal_alpha = optimal_result[0]
min_p_value = optimal_result[1]
fisher_statistic_val = optimal_result[2]

# Print optimal result details
print(f"Optimal alpha value: {optimal_alpha} and Fisher's p-value: {min_p_value}")
print(f"Fisher's Exact Test statistic for optimal alpha: {fisher_statistic_val}")

# Print all Fisher results for review
for idx, (statistic, p_value) in enumerate(fisher_results):
    print(f"Alpha: {alpha_values[idx]}, Fisher Statistic: {statistic}, P-value: {p_value}")


# In[176]:


import statsmodels
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt

# Generate alpha values from 0.05 to 0.25 in steps of 0.01
alpha_values = np.arange(0.05, 0.5, 0.01)

results = []  # Initialize results as a list

for alpha in alpha_values:
    # Perform the FDR correction
    FDR = multipletests(p_values,
                        alpha=alpha,
                        method='fdr_bh',
                        maxiter=1,
                        is_sorted=False,
                        returnsorted=False)

    # Extract q-values
    q_values = FDR[1]

    # Find significant q-values with the threshold of the current alpha value
    significant_q = np.where(q_values <= alpha)[0]

    # The significant column indexes are the experimental functional coordinates
    experimental_fs_coordinates = significant_q

    # A: Functional regions intersecting with ground truth
    shared_fs = [item for item in experimental_fs_coordinates if item in relics_fs_coordinates]

    # Experimental non-functional regions
    experimental_non_functional = [i for i in range(1536) if i not in experimental_fs_coordinates]

    # D: Non-functional coordinates in computed regions that are also in ground truth regions
    shared_non_fs = [item for item in experimental_non_functional if item in relics_non_functional]

    # C: Experimentally non-functional regions that are actually ground truth regions
    relics_exp_non_functional = [item for item in experimental_non_functional if item in relics_fs_coordinates]

    # B: Experimentally functional regions that are insignificant RELICS regions
    non_relics_exp_functional = [item for item in experimental_fs_coordinates if item in relics_non_functional]

    # Calculate A, B, C, D
    A = len(shared_fs)
    B = len(non_relics_exp_functional)
    C = len(relics_exp_non_functional)
    D = len(shared_non_fs)

    left = A + C
    right = B + D

    # Normalize A and D
    A = len(shared_fs) / left if left > 0 else 0  # Prevent division by zero
    D = len(shared_non_fs) / right if right > 0 else 0  # Prevent division by zero
   
    # Append to results for further analysis
    results.append((alpha, A, D))  # Append result

# Extract values for plotting
alpha_vals = [result[0] for result in results]  # Extract alpha values from results
A_values = [result[1] for result in results]     # Extract A values from results
D_values = [result[2] for result in results]     # Extract D values from results

# Plot the Fisher statistics against alpha values
plt.figure(figsize=(10, 5))

# Plotting A values
plt.plot(alpha_vals, A_values, marker='o', linestyle='-', color='b', label='Sensitivity')

# Plotting D values
plt.plot(alpha_vals, D_values, marker='o', linestyle='-', color='r', label='Specificity')

# Find intersection point
intersection_index = np.argmin(np.abs(np.array(A_values) - np.array(D_values)))
intersection_alpha = alpha_vals[intersection_index]
intersection_A = A_values[intersection_index]
intersection_D = D_values[intersection_index]

# Annotate the intersection point
plt.annotate(f'α = {intersection_alpha:.2f}', 
             (intersection_alpha, intersection_A), 
             textcoords="offset points",
             xytext=(0,30),  # Offset the label above the intersection point
             ha='center',
             fontsize=20,
             color='black',
             arrowprops=dict(arrowstyle='->', color='black'))

plt.title("Sensitivity vs. Specificity")
plt.xlabel("Alpha Values")
plt.ylabel("Score")
plt.grid()
plt.xticks(np.arange(0.05, 0.5, 0.05))

# Adding a legend to differentiate between A and D values
plt.legend()

# Show the plot
plt.show()


# In[ ]:




