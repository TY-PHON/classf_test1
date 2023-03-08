import torch
import os
from dataset import FoodDataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model1 import Classifier
import torch.nn as nn
from tqdm.auto import tqdm


#data augmentations 
#test aug
test_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])
#train aug
train_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

if __name__ == '__main__':

    batch_size = 64
    _dataset_dir = "../data"

    train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm = train_tfm)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm = test_tfm)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # "cuda" when available
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device('cpu')

    n_epochs = 4
    patience = 300 #if no improvement in 'patience' epochs, stop

    #create model, and put it on the device
    model = Classifier()
    model.to('cpu')

    print("model loaded!")

    #we use cross-entropy as the measurement of performance
    criterion = nn.CrossEntropyLoss()

    #Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    print("optimizer ok!")

    #Initialize trackers
    stale = 0 
    best_acc = 0

    for epoch in range(n_epochs):
        ###don't understand
        model.train()
        print("model train")

        train_loss = []
        train_accs = []

        for batch in train_loader:
            imgs, labels = batch

            print("bb")
            ###forward the data 
            logits = model(imgs.to(device))
            print("ok0")

            #calculate the cross-entropy loss
            loss = criterion(logits, labels.to(device))
            print("ok1")

            #?clear out the  gradients in the previous step
            optimizer.zero_grad()
            print("ok2")

            loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            train_loss.append(loss.item())
            train_accs.append(acc)

            ##print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        ### how to calculate?
        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)

        #print the information
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        ###make sure the model is in eval mode
        model.eval()
        
        valid_loss = []
        valid_acc = []

        for batch in tqdm(valid_loader):
            imgs, labels = batch

            #we don't need gradient
            with torch.no_grad():
                logits = model(imgs.to(device))
            
            loss = criterion(logits, labels.to(device))

            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            valid_loss.append(loss.item())
            valid_acc.append(acc)
        
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_acc) / len(valid_acc)

        print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        #output
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt","a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        #find the best and early stop
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break
    
    #test
    test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model_best = Classifier().to(device)
    model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
    model_best.eval()
    prediction = []
    with torch.no_grad():
        for data,_ in test_loader:
            test_pred = model_best(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            prediction += test_label.squeeze().tolist()

def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_set)+1)]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)
