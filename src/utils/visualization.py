import gc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import traceback
from pynvml import *

def show_torch_memory():
    info_str = ''
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    info_str += (f'total    : {info.total / (1024 * 1024)} Mb\n')
    info_str += (f'free     : {info.free/ (1024 * 1024)} Mb\n')
    info_str += (f'used     : {info.used/ (1024 * 1024)} Mb\n')

    print(info_str)
    return info_str

class Demo:

    def __init__(self, imgs, ft, ft_std, img_size):

        self.ft = ft # NCHW
        self.ft_std = ft_std # N1HW
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = img_size
        show_torch_memory()

    def plot_img_pairs(self, fig_size=3, alpha=0.45, scatter_size=70):
        print('feat_size:', self.ft.size(), 'feat_std_size:', self.ft_std.size(), '| img_size:', self.img_size)
        
        fig, axes = plt.subplots(1, self.num_imgs, figsize=(fig_size*self.num_imgs, fig_size))

        plt.tight_layout()

        for i in range(self.num_imgs):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('source image')
            else:
                axes[i].set_title('target image')

        num_channel = self.ft.size(1)
        cos = nn.CosineSimilarity(dim=1)
        
        def onclick(event):
            try:
                if event.inaxes == axes[0]:
                    with torch.no_grad():
                        
                        x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
    
                        src_ft = self.ft[0].unsqueeze(0)
                        src_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(src_ft)
                        src_vec = src_ft[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1
    
                        del src_ft
                        gc.collect()
                        torch.cuda.empty_cache()
    
                        trg_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(self.ft[1:])
                        trg_ft_std = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(self.ft_std[1:].cpu()) # in cpu since here I only acess a cell in it, no calculation over it
                        cos_map = cos(src_vec.cpu(), trg_ft.cpu()).cpu().numpy()  # N, H, W
                        
                        del trg_ft
                        gc.collect()
                        torch.cuda.empty_cache()
    
                        axes[0].clear()
                        axes[0].imshow(self.imgs[0])
                        axes[0].axis('off')
                        axes[0].scatter(x, y, c='r', s=scatter_size)
                        axes[0].set_title('source image')
    
                        for i in range(1, self.num_imgs):
                            max_yx = np.unravel_index(cos_map[i-1].argmax(), cos_map[i-1].shape)
                            max_cos_val = cos_map.max()
                            max_cos_std = trg_ft_std[0, 0, max_yx[1].item(), max_yx[0].item()]
                            axes[i].clear()
    
                            heatmap = cos_map[i-1]
                            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                            max_normed_val = (heatmap.max() / heatmap.sum()) * 100
                            
                            axes[i].imshow(self.imgs[i])
                            axes[i].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
                            axes[i].axis('off')
                            axes[i].scatter(max_yx[1].item(), max_yx[0].item(), c='r', s=scatter_size)
                            axes[i].annotate(f'{max_normed_val}', xy=(max_yx[1].item(), max_yx[0].item()),
                                             xytext=(0, 4), textcoords='offset points', ha='center',
                                             va='bottom')
                            axes[i].annotate('%.3f' % max_cos_val, xy=(max_yx[1].item(), max_yx[0].item()),
                                             xytext=(0, -5), textcoords='offset points', ha='center',
                                             va='bottom')
                            axes[i].annotate('%.3f' % max_cos_std, xy=(max_yx[1].item(), max_yx[0].item()),
                                             xytext=(0, -14), textcoords='offset points', ha='center',
                                             va='bottom')
                            axes[i].set_title('target image')
    
                        del cos_map
                        del heatmap
                        gc.collect()
            except Exception as e:
                e_str = str(traceback.format_exc())
                e_str = '\n'.join(e_str[i:i+80] for i in range(0, len(e_str), 80))
                axes[0].imshow(np.full((500,500), 0))
                axes[0].annotate(f'{e_str}', xy=(250,400),
                                             xytext=(0, 0), textcoords='offset points', ha='center',
                                             va='bottom', backgroundcolor="w", fontsize=8)
                # print(e)
                # print(traceback.trace())

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        