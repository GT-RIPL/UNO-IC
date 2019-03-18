import numpy as np
import matplotlib.pyplot as plt

num_classes = 11
num_images = 50

pred = np.random.randint(0,num_classes,(100,100,num_images))
pred_var = np.random.rand(100,100,num_images)

gt = np.random.randint(0,num_classes,(100,100,num_images))

rand = np.random.rand(100,100,num_images)

pred = gt.copy()
pred[rand>pred_var] = 0

print(pred.shape,pred_var.shape,gt.shape)

per_class_gt_var = {}
steps = 10
ranges = list(zip([1.*a/steps for a in range(steps+2)][:-2],
                  [1.*a/steps for a in range(steps+2)][1:]))


per_class_match_var = {r:{c:{} for c in range(num_classes)} for r in ranges}
overall_match_var = {r:{} for r in ranges}

print(overall_match_var)
in_range = 0
for r in ranges:
    # for each confidence range, tally correct labels and average confidence level
    low,high = r
    idx_pred_gt_match = (pred==gt) # everywhere correctly labeled
    idx_pred_var_in_range = (low<=pred_var)&(pred_var<high) # everywhere with specified confidence level

    in_range += np.sum(idx_pred_var_in_range[:])

    average_pred_var = np.sum(pred_var[idx_pred_var_in_range][:])/np.sum(idx_pred_var_in_range[:])
    average_obs_var = np.sum((idx_pred_gt_match&idx_pred_var_in_range)[:])/np.sum(idx_pred_var_in_range[:])

    overall_match_var[r]['pred'] = average_pred_var
    overall_match_var[r]['obs'] = average_obs_var

    for c in range(num_classes):
        # for each class, record number of correct labels for each confidence bin
        # for each class, record average confidence for each confidence bin 

        low,high = r
        idx_pred_gt_match = (pred==gt)&(pred==c) # everywhere correctly labeled to correct class
        idx_pred_var_in_range = (low<=pred_var)&(pred_var<high) # everywhere with specified confidence level

        average_pred_var = np.sum(pred_var[idx_pred_var_in_range][:])/np.sum(idx_pred_var_in_range[:])
        average_obs_var = np.sum((idx_pred_gt_match&idx_pred_var_in_range)[:])/np.sum(idx_pred_var_in_range[:])

        per_class_match_var[r][c]['pred'] = average_pred_var
        per_class_match_var[r][c]['obs'] = average_obs_var

print(in_range)
print(overall_match_var)

x = [overall_match_var[r]['pred'] for r in overall_match_var.keys()]
y = [overall_match_var[r]['obs'] for r in overall_match_var.keys()]

print(x,y)

plt.figure()
plt.plot(x,y)
plt.show()


exit()

# per_class_gt_var = 

print(per_class_gt_var)


plt.figure()
plt.subplot(2,4,1)
plt.imshow(gt[:,:,0])
# plt.subplot(2,4,2)
# plt.imshow(gt_var[:,:,0])
plt.subplot(2,4,3)
plt.imshow(pred[:,:,0])
plt.subplot(2,4,4)
plt.imshow(pred_var[:,:,0])


plt.subplot(2,4,5)
plt.imshow(pred[:,:,0]==gt[:,:,0])
# plt.subplot(2,4,6)
# plt.imshow(pred_var[:,:,0])


plt.show()