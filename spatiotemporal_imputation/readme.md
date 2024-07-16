# installation
1- Conda create a new environment, I create 'PyG' environment with Python 3.9.
2- You should install PYTORCH first, as reconmmended in page 'https://pytorch.org/get-started/previous-versions/', conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 cpuonly -c pytorch.
3- Then, install those packages IN ORDER: ①torch_scatter②torch_sparse③torch_cluster④torch_spline_conv(Step3)⑤torch-geometric(Step4)，You can download the wheel from website 'https://data.pyg.org/whl/', and the version should match your pytorch version, for me, it's pytorch==2.1.0.
4- Finally, pip install torch_geometric, as recommended in 'https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html', because I use CPU, it will be different if u use GPU, CUDA, maybe, so, double check it.
5- If u use jupyter lab, remenber pip install ipykernel to change your environment.


# result
airquality: 
- ST-transformer+additional st covariates: 9.2
- ST-transformer: 11.6
