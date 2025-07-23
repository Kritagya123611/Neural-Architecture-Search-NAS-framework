from dataset import loaders

trainloader, testloader = loaders()
# trainloader aur testloader ko import karne ke liye
for images,labels in trainloader:
    print("image shape:",images.shape)  # images ka shape dekhne ke liye
    print("label shape:",labels.shape)  # labels ka shape dekhne ke liye
    break  # sirf pehla batch dekhne ke liye