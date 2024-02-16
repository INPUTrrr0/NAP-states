import numpy as np
from torchvision import datasets, transforms
import torch
import pandas as pd
#from google.colab import files
import seaborn as sns
import matplotlib.pyplot as plt

import onnx
from onnx2pytorch import ConvertModel
from tqdm import tqdm

import os
import json

RELU_NUM=6
TRAIN_BS = 30
TEST_BS = 30
USE_CUDA = torch.cuda.is_available()
DLD_DATA = True


def plot_neuron(nonzeros_relu, activations_relu, layer):
    layer_validneurons=[]
    layer_states={}
    # Plot histograms for each label
    for neuron_index, neuron in tqdm(enumerate(nonzeros_relu)): #TODO: here
      #print(neuron)
      fig, axes = plt.subplots(nrows=10, ncols=1, figsize=(10, 20))
      fig.suptitle(f'Neuron {neuron} in {layer}', fontsize=16,y=1)
      fig.subplots_adjust(top=0.95)
      neuron_max=max(max(x) for x in activations_relu[f'neuron{neuron}'].values()) #TODO: here
      count=0

      for i in range(10):  # Assuming 10 labels
        ax = axes[i]
        layer_states[neuron]=[[] for i in range(10)]
        data=activations_relu[f'neuron{neuron}'][f'label{i}'] #TODO: here
      
        if max(data)==0: #or np.median(data)==0:
            continue
            #print(f'mean of label {i}: {np.mean(list_of_lists[i])} \n std of label {i}: {np.std(list_of_lists[i])}')
        # temp_fig = plt.figure()
        # counts, bin_edges, patches = plt.hist(data, bins=20)  # Adjust the number of bins as necessary
        # plt.close(temp_fig)
        #if ((len(data)-data.count(0))/len(data))<0.05 or np.median(data)==0:#((len(data)-data.count(0))/len(data))<0.05:#median!=0:#len(counts)<3:#counts[0] > max(counts[1:]):  # Check if the first bin count is higher than the rest
          #continue
        # data.remove(max(data))
        layer_states[neuron][i]=data
        #print(layer3_states[neuron])
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        max_val=max(data)
        count=count+1
        #print(f"neuron:{neuron}, label{i}: range:{mean-std}, {mean+std},")
        ax.hist(data, bins=20, color='red', alpha=0.5, label=f'Neuron {neuron_index+1}') #width=0.1
        ax.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
        ax.axvline(max_val, color='r', linestyle='dotted', linewidth=2, label=f'Max: {max_val:.2f}')
        ax.axvline(median, color='g', linestyle='dotted', linewidth=2, label=f'Median: {median:.2f}')
        ax.axvline(mean - std, color='b', linestyle='dashed', linewidth=1, label=f'Std Dev: {std:.2f}')
        ax.axvline(mean + std, color='b', linestyle='dashed', linewidth=1)
        ax.set_xlim([-1, neuron_max])
        ax.legend()

        ax.set_title(f'Label {i} (nonzero datapoints: {(len(data)-data.count(0))}/{len(data)})')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
    
        if count>0:
            layer_validneurons.append(neuron)
            axes[0].legend()
            plt.tight_layout()
            save_pth = os.path.join("/mfs1/huang/Sophie/NAP-states/mnistfc_256x6_states", 'non_zero_neurons', f'layer{layer}_states')
            os.makedirs(save_pth, exist_ok=True)
            plt.savefig(f'{save_pth}/{neuron}')
        #plt.show()
        plt.close()
    
    os.makedirs(os.path.join("/mfs1/huang/Sophie/NAP-states/mnistfc_256x6_states", 'non_zero_neurons', f'relu{layer}'), exist_ok=True)
    with open(os.path.join("/mfs1/huang/Sophie/NAP-states/mnistfc_256x6_states", 'non_zero_neurons', f'relu{layer}',f"filtered_relu{layer}_activations.json"), "w") as fp:
      json.dump(layer_states, fp)


def main():
    onnx_model = onnx.load(f"{os.getcwd()}/mnist-net_256x6.onnx")
    pytorch_model = ConvertModel(onnx_model)

    device = torch.device("cuda" if USE_CUDA else "cpu")
    train_kwargs = {'batch_size': TRAIN_BS, 'shuffle': False}
    test_kwargs = {'batch_size': TEST_BS, 'shuffle': False}
    # if USE_CUDA:
    #     cuda_kwargs = {'num_workers': 1,
    #                    'pin_memory': True,
    #                    'shuffle': True}
    #     train_kwargs.update(cuda_kwargs)
    #     test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    train_set = datasets.MNIST('./data', train=True, download=DLD_DATA,
                            transform=transform)
    test_set = datasets.MNIST('./data', train=False, download=DLD_DATA,
                            transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, **test_kwargs)


    single_datapoint = next(iter(test_loader))
    print(f'{"train datapoint size",single_datapoint[0].shape}')
    print(f'{"train target size", single_datapoint[1].shape}')
    
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1) #2 rows, 3 columns. 1 2 3, (next row) 4 5 6
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    
    print(len(example_data[0][0][0]))
    print(pytorch_model(example_data[0]).argmax())
    print(pytorch_model)
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook


    # Attach the hook to the 1st layer
    pytorch_model.Relu_17.register_forward_hook(get_activation('relu1'))
    pytorch_model.Relu_19.register_forward_hook(get_activation('relu2'))
    pytorch_model.Relu_21.register_forward_hook(get_activation('relu3'))
    pytorch_model.Relu_23.register_forward_hook(get_activation('relu4'))
    pytorch_model.Relu_25.register_forward_hook(get_activation('relu5'))
    pytorch_model.Relu_27.register_forward_hook(get_activation('relu6'))
    
    if not os.path.exists(os.path.join("/mfs1/huang/Sophie/NAP-states/mnistfc_256x6_states", 'non_zero_neurons')):
        # Create list to store label
        labels = []

        dict_relu = dict()
        for relu_num in range(RELU_NUM):
            dict_relu[f'relu{relu_num}'] = []

        # Pass data through the network
        for batch_idx, (example_data, example_targets) in tqdm(enumerate(test_loader)):
            for i,t in zip(example_data, example_targets):
                output = pytorch_model(i)
                # record the neuron output
                for layer in range(RELU_NUM):
                    neuron_output = activation[f'relu{layer + 1}']
                    dict_relu[f'relu{layer}'].append(neuron_output)
                
                labels.append(t.item())
        
        os.makedirs(os.path.join("/mfs1/huang/Sophie/NAP-states/mnistfc_256x6_states", 'non_zero_neurons'), exist_ok=True)
        # Record all the non-zero neurons
        non_zero_neurons = []
        for layer in dict_relu:
            all_images = torch.cat(dict_relu[layer], dim=0)
            non_zero_neurons = []
            non_zero_activation = dict()
            for neuron_num in range(all_images.shape[1]):
                neuron_activation = all_images[:, neuron_num]
                if torch.count_nonzero(neuron_activation) > 0:
                    non_zero_neurons.append(neuron_num)
                    non_zero_activation[f'neuron{neuron_num}'] = dict()

                    for i in range(10):
                        non_zero_activation[f'neuron{neuron_num}'][f'label{i}'] = []

                    for k in range(len(labels)):
                        label = labels[k]
                        non_zero_activation[f'neuron{neuron_num}'][f'label{label}'].append(neuron_activation[k].item())
            
            os.makedirs(os.path.join("/mfs1/huang/Sophie/NAP-states/mnistfc_256x6_states", 'non_zero_neurons', layer), exist_ok=True)
            with open(os.path.join("/mfs1/huang/Sophie/NAP-states/mnistfc_256x6_states", 'non_zero_neurons', layer, 'nonzero.txt'), 'w') as f:
                # write elements of list
                json.dump(non_zero_neurons, f, indent=4)
                f.close()
            
            with open(os.path.join("/mfs1/huang/Sophie/NAP-states/mnistfc_256x6_states", 'non_zero_neurons', layer, 'activation.json'), 'w') as f1:
                # write elements of dictionary
                json.dump(non_zero_activation, f1, indent=4)
                f1.close()
            
            import re
            curr_layer = re.findall("\d+", layer)[0]
            plot_neuron(non_zero_neurons, non_zero_activation, curr_layer)

    else:
        for relu_num in range(RELU_NUM):
            with open(os.path.join("/mfs1/huang/Sophie/NAP-states/mnistfc_256x6_states", 'non_zero_neurons', f'relu{relu_num}', 'activation.json'), 'r') as f:
                activations_relu = json.load(f)
            
            with open(os.path.join("/mfs1/huang/Sophie/NAP-states/mnistfc_256x6_states", 'non_zero_neurons', f'relu{relu_num}', 'nonzero.txt'), 'r') as f1:
                non_zero_neurons = json.load(f1)
                
            plot_neuron(non_zero_neurons, activations_relu, relu_num)
    
    
if __name__ == "__main__":
    main()

