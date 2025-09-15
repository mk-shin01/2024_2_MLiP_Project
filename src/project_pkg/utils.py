"""General utilities extracted from the original notebook."""


# ---- cell 1 ----
with h5py.File('/content/Galaxy10_DECals.h5', 'r') as F:
    images = np.array(F['images']) # Select only few examples [0:5000]
    labels = np.array(F['ans'])

print('Data loaded successfully')

class_names = ['Disturbed', 'Merging', 'Round_Smooth',
               'In-between_Round_Smooth', 'Cigar_Shaped_Smooth', 'Barred_Spiral',
               'Unbarred_Tight_Spiral', 'Unbarred_Loose_Spiral', 'Edge-on_without_Bulge',
               'Edge-on_with_Bulge']

# ---- cell 2 ----
img = None
plt.ion()
samples = np.random.randint(0, labels.shape[0], size=5)
print('Images index to display:', samples)
for i in samples:
    img = plt.imshow(images[i])
    plt.title('Image {}: Class {}'.format(i, labels[i]))
    plt.draw()
    plt.pause(2.)
plt.close('all')

# ---- cell 3 ----
dataset_dir = '/content/Galaxy10/'

# List models from:
model_name = 'mobilenetv2'  # 모델 이름을 'mobilenetv2'로 변경

training_epochs = 40
schedule_steps = 5
learning_rate = 0.0005
batch_size = 64

# ---- cell 4 ----
import torch
import torchvision
import torchsummary
import sys

# ---- cell 5 ----
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Model initialization
if model_name == 'mobilenetv2':  # 'mobilenetv2' 모델을 사용하도록 변경
    model_ft = MobileNetV2(num_classes=10).cuda()  # MobileNetV2 모델로 초기화

# 모델 파라미터 수 출력
pytorch_total_params = sum(p.numel() for p in model_ft.parameters())
print(f"Number of parameters: {pytorch_total_params}")
if int(pytorch_total_params) > 5000000:
    print('Your model has the number of parameters more than 5 millions..')
    sys.exit()

model_ft = model_ft.to(device)

# 모델 요약 정보
torchsummary.summary(model_ft, (3, 224, 224))

