
# Write functions so vectorized operations can work on Series and Arrays. 


```
# Get index array if numpy or pandas series
idx_arr = np.arange(len(srf_arr)) if isinstance(srf_arr, np.ndarray) else srf_arr.index 
idx_arr[srf_arr > 10] 
```

This means all dataframes must be unzipped into series when adding to functions. Here's an example:

```
def fx(a_arr, b_arr):
	return a_arr + b_arr

c_arr = fx(*df.loc[idx_arr, ['col_a', 'col_b']].T.values)
```
