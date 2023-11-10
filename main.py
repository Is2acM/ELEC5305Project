'''
this script is for the training code of ELEC5305 Research Project.

-------------------------------------------
AUTHOR: 
Written by Yuefeng Ma in Nov 3, 2023. 

-------------------------------------------
USAGE:

--epochs -e: Specified value of epochs to be run. Default is 1 epoch

--cuda: Running the training process in cuda framework. Default is False

e.g. 
>> python main.py 
This will run the program on CPU in 1 epoch

>> python main.py --cuda -e 20
To perform Training, this will run the training process with number of 
epochs specified, and train the network with cuda framework. (Here is 20 epochs)

-------------------------------------------
SPECS: 
    GPU: RTX 3070Ti 8GB  
    CPU: 13700KF  
    RAM: 32GB  
    CUDA: 11.7  
    Pytorch Version: 2.0.1  
    Python: 3.9.0
'''

## Import the packages
import time
import torch
import torch.optim as optim
from torch import nn
from tqdm.auto import tqdm
from datasets import train_loader, valid_loader, classes
from utils import save_model, save_plots, create_logger, parse_args, saveCM

## Uncomment to decide which network to train
## Import Resnet50 structure
# from Resnet50 import Network 
## Import ViT structure
from ViT import Network
## Import CNN structure
# from CNN import Network

## Define Train and Validate functions ##

def train(model, trainloader, optimizer, criterion, scheduler):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # calculate the loss
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # calculate the accuracy
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # backpropagation
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # Apply Scheduler
    if type(scheduler).__name__ != 'NoneType':
            before_step = optimizer.param_groups[0]["lr"]
            scheduler.step()
            after_step = optimizer.param_groups[0]["lr"]
            print(f"Learning Rate: {before_step:.4f} -> {after_step :.4f}")
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc

def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    y_pred = []
    y_true = []
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()
            y_pred.extend(preds.cpu())
            y_true.extend(labels.cpu())
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc, y_pred, y_true


## Initialization ##
args = vars(parse_args()) 
epochs = args['epochs']
device = ('cuda' if args['cuda'] else 'cpu')
# device = 'mps' # Uncomment this if running on Mac
logger = create_logger()
logger.info('using args:')
logger.info(args)
logger.info(f"Computation device: {device}\n")

## Specify Model
model = Network(train = True) # Model set to training mode
model = model.to(device)
logger.info(model)            # Record information of the model (Structures)
# total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")


## Optimizer and Loss Definition ##
## Resnet50 ##
# lr = 0.001
# criterion = nn.CrossEntropyLoss()
# # Mel, MFCCs hyper-parameters
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
# scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.1) 
# # Chroma hyper-parameters
# optimizer = optim.Adam(model.parameters(), lr = lr) 
# scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1) 

## ViT ##
# # Mel, MFCCs hyper-parameters
lr = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.2) 

# # Chroma hyper-parameters
# lr = 0.001
# optimizer = optim.Adam(model.parameters(), lr=lr) # Chroma
# scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=3, gamma=0.1) # Chroma
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9) 
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10], gamma = 0.1)



## CNN ##
# lr = 0.001 # Mel, MFCCs and Chroma, Learning Rate
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.1) # Mel and MFCCs
# scheduler = None # Chroma



## training process. ##
if __name__ == '__main__':
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    y_pred, y_true = [],[]
    ## start the training ##
    start_time = time.time()
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion, scheduler)
        valid_epoch_loss, valid_epoch_acc, y_pred, y_true = validate(model, valid_loader,  
                                                    criterion)
        # Record the loss and accuracy at each epoch
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-'*50)
    end_time = time.time()
    total_time = end_time - start_time # Program Timer
    ## Training End ##

    # Record infomation of the training
    logger.info(f'Total training time: {total_time:.2f} seconds')
    logger.info(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
    logger.info((f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}"))
    # save the trained model weights
    save_model(model)
    # save the loss and accuracy plots
    save_plots(train_acc, valid_acc, train_loss, valid_loss)
    saveCM(y_true, y_pred, classes)
    logger.info('TRAINING COMPLETE')
    # ==================================
