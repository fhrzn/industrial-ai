from tqdm.auto import tqdm
from torch.optim import AdamW
from transformers import get_scheduler
import torch
import os
import evaluate

def train(model, model_name, trainloader, valloader=None, 
            config=None, device=torch.device('cpu'),
            output_path=None):

    #######################
    # training properties #
    #######################
    # optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['text_model_learning_rate'],
        weight_decay=0.001
    )
    # epoch & step
    n_epoch = config['num_epoch']
    num_train_steps = n_epoch * len(trainloader)
    # LR scheduler
    scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    # early stop
    n_early_stop = config['early_stop']
    val_loss_min = torch.inf
    # history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': n_epoch
    }
    # metric
    metric = evaluate.load('accuracy')
    # tqdm
    progress_bar = tqdm(range(num_train_steps))
    # output path
    if output_path is None:
        output_path = './artifact/'
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)

    #################
    # training loop #
    #################
    for e in range(n_epoch):
        
        # train mode
        model.train()

        train_loss = 0
        train_acc = 0

        for batch in trainloader:
            outputs = model(**batch.to(device))
            
            # loss
            loss = outputs.loss
            train_loss += loss
            loss.backward()

            # acc
            preds = torch.argmax(outputs.logits, dim=-1)            
            train_acc += metric.compute(predictions=preds, references=batch.labels)['accuracy']

            # step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # log history
        history['train_loss'].append(train_loss / len(trainloader))
        history['train_acc'].append(train_acc / len(trainloader))

        if valloader is None:
            continue
        
        # eval mode
        model.eval()

        val_loss = 0
        val_acc = 0

        for batch in valloader:
            with torch.no_grad():
                outputs = model(**batch.to(device))

            # loss
            loss = outputs.loss
            val_loss += loss

            # acc
            preds = torch.argmax(outputs.logits, dim=-1)
            val_acc += metric.compute(predictions=preds, references=batch.labels)['accuracy']
            
        # log history
        history['val_loss'].append(val_loss / len(valloader))
        history['val_acc'].append(val_acc / len(trainloader))

        # print log        
        print_log(e, history, progress_bar, config['log_every'])

        # save model if val loss decrease
        if val_loss / len(valloader) <= val_loss_min:
            torch.save(model.state_dict(), output_path + model_name + '.pt')
            val_loss_min = val_loss / len(valloader)
            es_trigger = 0
        else:
            progress_bar.write(f'[WARN] Vloss did not improved ({val_loss_min:.3f} --> {val_loss / len(valloader):.3f})')
            es_trigger += 1
        
        # early stop
        if es_trigger >= n_early_stop:
            progress_bar.write(f'Early stopped at Epoch-{e + 1}')
            # updte epochs history
            history['epochs'] = e+1
            break

    return model, history
    

def test(model, testloader, device=torch.device('cpu')):
    
    metric = evaluate.load('accuracy')
    
    # eval mode
    model.eval()

    test_loss = 0
    test_acc = 0

    for batch in testloader:
        with torch.no_grad():
            outputs = model(**batch.to(device))

        # loss
        loss = outputs.loss
        test_loss += loss

        # acc
        preds = torch.argmax(outputs.logits, dim=-1)
        test_acc += metric.compute(predictions=preds, references=batch.labels)['accuracy']

    return test_loss / len(testloader), test_acc / len(testloader)



def print_log(cur_epoch, history, progressbar, log_every=1):
    if (cur_epoch + 1) % log_every == 0:
        progressbar.write(f"TLoss: {history['train_loss'][-1]:.3f} | TAcc: {history['train_acc'][-1]:.3f} | VLoss: {history['val_loss'][-1]:.3f} | VAcc: {history['val_acc'][-1]:.3f}")