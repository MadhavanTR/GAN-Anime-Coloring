# -*- coding: utf-8 -*-
# Number of sketches to predict the colors. 
k = 10
test_skets = []
idxs = np.random.randint(2335, TOTAL_IMAGES - 2335, k)

for sket, img in zip(sketch_paths[idxs], img_paths[idxs]):
    test_skets.append(np.array(Image.open(sket).convert('RGB')))

# Scaling such that all values come into the range of [-1, 1].
test_skets = np.array(temp_skets, dtype='float32')/127.5 - 1

# Predicting the colors for the simple line sketches. 
pred = g_model.predict(test_skets)

# Rescaling back into the range of [0, 255] from [-1, 1].
pred = (pred+1)/2.0
test_skets = (test_skets+1)/2.0

# Plotting the generated colored anime images.
f, a = plt.subplots(k, 2, figsize=(12,60)); a = a.flatten()
idx = 0

for sket, pic in zip(temp_skets, pred):
  a[idx].imshow(sket); a[idx].axis('off')
  a[idx+1].imshow(pic); a[idx+1].axis('off')
  idx += 2

plt.subplots_adjust(wspace=.1, hspace=.1)
plt.show()
