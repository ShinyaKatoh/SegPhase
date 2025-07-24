# SegPhase : Development of Arrival Time Picking Models for Japan’s Seismic Network Using the Hierarchical Vision Transformer  
(https://earth-planets-space.springeropen.com/articles/10.1186/s40623-025-02249-y)
## Model Structure  
![Test Image 1](/images/SegPhase-1.png)　　
## How to use  
If you want to input from WIN files in [main.py](/main.py) or [main.ipynb](/main.ipynb),  
you need to install the WIN system.  
Please refer to the following site to download the [WIN](https://wwweic.eri.u-tokyo.ac.jp/WIN/Eindex.html) system.  
We also have test data for npz files as well. When you want to use npz files, please uncomment out accordingly.
  
If you want to run it in google colab,  
please use [main_for_GoogleColab.ipynb](/main_for_GoogleColab.ipynb)  
(However, input from win files is not possible.)

For the 100 Hz model, the input to the model is (Batch size, channel, data length) = (Batch size, 3, 3000), and for the 250 Hz model, it is (Batch size, channel, data length) = (Batch size, 3, 7500). The order of the channel data is UD, NS, EW. For the M01 model, the input is (Batch size, channel, data length) = (Batch size, 1, 3000).
