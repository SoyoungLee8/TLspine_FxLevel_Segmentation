from scipy.ndimage import label
import numpy as np
import torch
import cv2


def refine_position(batch_yhat):
    """
    batch_yhat : B x C x H x W only after softmax
    result : B x 1 x H x W
    index : 0, 1(L), 2(T), 3(S)
    """
    batch_yhat = torch.argmax(batch_yhat,1).unsqueeze(1) # B x 1 x H x W
    batch_yhat_1 = batch_yhat.clone()
    batch_yhat_1[batch_yhat_1!=0] = 1
    result = np.zeros_like(batch_yhat_1.cpu().detach().numpy())
    # result = np.zeros_like(batch_yhat_1.cpu().detach().numpy())
    kernel = np.ones((5,5), np.uint8)
    for i in range(len(batch_yhat_1)):
        batch_yhat_2 = cv2.morphologyEx((batch_yhat_1[i,0].cpu().detach().numpy()).astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=5)
        spines, _ = label(batch_yhat_2)
        for j in range(1, len(np.unique(spines))): # 0(=background) 빼고 iter
            spine = spines.copy()
            # body 1개만 1로 만들기
            spine[spine!=j] = 0
            spine[spine!=0] = 1
            unique, counts = np.unique(spine*batch_yhat[i,0].cpu().detach().numpy(),return_counts=True)
            if 3 in unique:
                idx = list(unique).index(3)
                if counts[idx]/np.sum(counts[1:]) >= 0.3:
                    spine = spine*3
                else:
                    spine = spine*batch_yhat[i,0].cpu().detach().numpy()
            elif 1 in unique:
                idx = list(unique).index(1)
                if counts[idx]/np.sum(counts[1:]) >= 0.3:
                    spine = spine*1 # L spine
                else:
                    spine = spine*2 # T spine
            elif 2 in unique:
                spine = spine*2
            result[i,0] += spine
        result[i,0] = refine5L(result[i,0])
    return result

def refine5L(result):
    '''
    input : H x W
    output : H x W
    When the number of L spine segment is 4: last segment of T spine convert L spine
    When the number of L spine segment is 6: first segment of L spine convert T spine
    '''
    sacrum = result.copy()
    sacrum[sacrum!=3] = 0
    
    Tspine = result.copy()
    Tspine[Tspine!=2] = 0
    
    Lspine = result.copy()
    Lspine[Lspine!=1] = 0
    
    lspines, _ = label(Lspine)
    if len(np.unique(lspines)) == 6:
        pass
    elif len(np.unique(lspines)) < 6:
        print('add L1')
        tspines, _ = label(Tspine)
        iter_num = 6 - len(np.unique(lspines))
        for i in range(iter_num):
            i += 1 # i = 1,2,3...
            i = (-1)*i # i = -1,-2,-3...
            L1 = np.unique(tspines)[i]
            Tspine[tspines==L1] = 1
    elif len(np.unique(lspines)) > 6:
        print('del T12')
        iter_num = len(np.unique(lspines)) - 6
        for i in range(iter_num):
            i += 1 # i = 1,2,3
            Lspine[lspines==i] = 2
    output = Tspine + Lspine + sacrum
    
    return output

def refine_fracture(batch_yhat, threshold=.3):
    """
    batch_yhat : only after softmax
    """
    batch_yhat = torch.argmax(batch_yhat,1).unsqueeze(1) # B x 1 x H x W
    batch_yhat_1 = batch_yhat.clone()
    batch_yhat_1[batch_yhat_1!=0] = 1
    result = np.zeros_like(batch_yhat_1.cpu().detach().numpy())
    
    # each batch
    kernel = np.ones((5,5), np.uint8)
    for i in range(len(batch_yhat_1)):
        batch_yhat_2 = cv2.morphologyEx((batch_yhat_1[i,0].cpu().detach().numpy()).astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=5)
        spines, _ = label(batch_yhat_2)
        for j in range(1, len(np.unique(spines))): # except 0(=background)
            spine = spines.copy()
            spine[spine!=j] = 0
            spine[spine!=0] = 1
            unique, counts = np.unique(spine*batch_yhat[i,0].cpu().detach().numpy(),return_counts=True)
            if 2 in unique:
                idx = list(unique).index(2)
                if counts[idx]/np.sum(counts[1:]) >= threshold:
                    spine = spine*2
            else:
                spine = spine*batch_yhat[i,0].cpu().detach().numpy()
            result[i,0] += spine
    return result