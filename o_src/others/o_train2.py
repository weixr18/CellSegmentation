# https://github.com/Hsankesara/DeepResearch/blob/master/UNet/run_unet.py


def train_step(inputs, labels, optimizer, criterion, unet, width_out, height_out):
    optimizer.zero_grad()
    # forward + backward + optimize
    outputs = unet(inputs)
    # outputs.shape =(batch_size, n_classes, img_cols, img_rows)
    outputs = outputs.permute(0, 2, 3, 1)
    # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
    m = outputs.shape[0]
    outputs = outputs.resize(m*width_out*height_out, 2)
    labels = labels.resize(m*width_out*height_out)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def get_val_loss(x_val, y_val, width_out, height_out, unet):
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).long()
    if use_gpu:
        x_val = x_val.cuda()
        y_val = y_val.cuda()
    m = x_val.shape[0]
    outputs = unet(x_val)
    # outputs.shape =(batch_size, n_classes, img_cols, img_rows)
    outputs = outputs.permute(0, 2, 3, 1)
    # outputs.shape =(batch_size, img_cols, img_rows, n_classes)
    outputs = outputs.resize(m*width_out*height_out, 2)
    labels = y_val.resize(m*width_out*height_out)
    loss = F.cross_entropy(outputs, labels)
    return loss.data


def train(unet, batch_size, epochs, epoch_lapse,
          threshold, learning_rate, criterion,
          optimizer, x_train, y_train, x_val, y_val,
          width_out, height_out):
    epoch_iter = np.ceil(x_train.shape[0] / batch_size).astype(int)
    t = trange(epochs, leave=True)
    for _ in t:
        total_loss = 0
        for i in range(epoch_iter):
            batch_train_x = torch.from_numpy(
                x_train[i * batch_size: (i + 1) * batch_size]).float()
            batch_train_y = torch.from_numpy(
                y_train[i * batch_size: (i + 1) * batch_size]).long()
            if use_gpu:
                batch_train_x = batch_train_x.cuda()
                batch_train_y = batch_train_y.cuda()
            batch_loss = train_step(
                batch_train_x, batch_train_y, optimizer, criterion, unet, width_out, height_out)
            total_loss += batch_loss
        if (_+1) % epoch_lapse == 0:
            val_loss = get_val_loss(x_val, y_val, width_out, height_out, unet)
            print("Total loss in epoch %f : %f and validation loss : %f" %
                  (_+1, total_loss, val_loss))
    gc.collect()


def main():
    width_in = 284
    height_in = 284
    width_out = 196
    height_out = 196
    PATH = './unet.pt'
    x_train, y_train, x_val, y_val = get_dataset(
        width_in, height_in, width_out, height_out)
    print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
    batch_size = 3
    epochs = 1
    epoch_lapse = 50
    threshold = 0.5
    learning_rate = 0.01
    unet = UNet(in_channel=1, out_channel=2)
    if use_gpu:
        unet = unet.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(unet.parameters(), lr=0.01, momentum=0.99)
    if sys.argv[1] == 'train':
        train(unet, batch_size, epochs, epoch_lapse, threshold, learning_rate,
              criterion, optimizer, x_train, y_train, x_val, y_val, width_out, height_out)
        pass
    else:
        if use_gpu:
            unet.load_state_dict(torch.load(PATH))
        else:
            unet.load_state_dict(torch.load(PATH, map_location='cpu'))
        print(unet.eval())
    plot_examples(unet, x_train, y_train)
    plot_examples(unet, x_val, y_val)

    pass


if __name__ == "__main__":
    main()
    pass
