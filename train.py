import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

class EmotionBodyDataset(Dataset):
    def __init__(self, image_paths, annotation_paths):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        annotation_path = self.annotation_paths[idx]

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Process annotations
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            bboxes = []
            categories = []
            emotions = []
            for line in lines:
                data = line.strip().split(',')
                bbox = [int(data[0]), int(data[1]), int(data[2]), int(data[3])]
                category = int(data[4])
                emotion = [float(data[5]), float(data[6]), float(data[7])]
                bboxes.append(bbox)
                categories.append(category)
                emotions.append(emotion)

        # Create sample dictionary
        sample = {'image': image, 'bboxes': bboxes, 'categories': categories, 'emotions': emotions}

        return sample


def train_model(model, train_loader, optimizer, criterion, device):
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].to(device)
        emotions = batch['emotions'].to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, emotions)

        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

if __name__ == '__main__':
    image_paths = [...]  # List of image paths
    annotation_paths = [...]  # List of annotation paths

    # Initialize dataset and dataloader
    dataset = EmotionBodyDataset(image_paths, annotation_paths)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # Initialize model, optimizer, and criterion
    model = EmotionBodyEncoder(num_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train model
    num_epochs = 10
    total_step = len(dataloader)
    for epoch in range(num_epochs):
        train_model(model, dataloader, optimizer, criterion, device)
