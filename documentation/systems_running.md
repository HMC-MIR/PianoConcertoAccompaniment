# Systems Running

We have online and offline processing for each of our systems. In offline processing, we compute the features for our queries and references. The features we're using are chroma stft features without centering and using L2 norm (we will call these `features` in this document). The only exception is OLTW from the MATCH jar file, which uses its own features. In online processing, we performa alignment between the query and reference. We will detail the process for onlines and offline processing for each system below.

## DTW

### Offline Processing

We calculate the `features` for the piano solo reference, which is the original piano track that is in sync with the orchestra, and save it to `{feature_dir}/pref_stft.npy`.

### Online Processing

DTW online processing is not actually online because DTW is intrinsically an offline algorithm.

We first verify our dicrectory structure. Then, we compute the `features` for the query input. We load in the `features` for the reference that we computed during offline processing. We calculate the cosine distance between the query and the reference. We then run subsequence DTW on this cost matrix, convert the path from frames to seconds, and save it to `{out_dir}/hyp.npy`.

## NOA/NOA_MONOTONIC

### Offline Processing

Same as DTW. We will reuse the features computed.

### Online Processing

Same as DTW except for the alignment section. However, since NOA doesn't support subsequence operations, we need to shift the reference features so that the starting point of the reference features correspond to start of the scenario. We then compute alignment path as normal. We can do this because NOA is an online algorithm and it terminates when we reach the end of the query.

## OLTW-GLOBAL

### Offline Processing

Same as DTW. We will reuse the features computed.

### Online Processing

Everything is the same as NOA.

## OLTW

### Offline Processing

For OLTW from MATCH jar file, we don't compute features during offline processing. Instead, we need to chop the reference audio such that we can pass it into the jar file. The chopped reference audio will be saved in the scenarios directory. The boundaries of chopping are determined from the scenario.info file.

### Online Processing

Check the System_OLTW.ipynb notebook. It's really complicated and we are sad and tired and we don't want to write this. If you're reading this, run.