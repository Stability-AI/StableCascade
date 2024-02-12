# Download Models

As there are many models provided, let's make sure you only download the ones you need.
The ``download_models.sh`` will make that very easy. The basic usage looks like this: <br>
```bash
bash download_models.sh essential variant bfloat16
```

**essential**<br>
This is optional and determines if you want to download the EfficientNet, Stage A & Previewer. 
If this is the first time you run this command, you should definitely do it, because we need it.

**variant**<br>
This determines which varient you want to use for **Stage B** and **Stage C**.
There are four options:

|                     | Stage C (Large) | Stage C (Lite) |
|---------------------|-----------------|----------------|
| **Stage B (Large)** | big-big         | big-small      |
| **Stage B (Lite)**  | small-big       | small-small    |


So if you want to download the large Stage B & large Stage C you can execute: <br> 
```bash
bash download_models.sh essential big-big bfloat16
```

**bfloat16** <br>
The last argument is optional as well, and simply determines in which precision you download Stage B & Stage C.
If you want a faster download, choose _bfloat16_ (if your machine supports it), otherwise use _float32_.

### Recommendation
If your GPU allows for it, you should definitely go for the **large** Stage C, which has 3.6 billion parameters.
It is a lot better and was finetuned a lot more. Also, the ControlNet and Lora examples are only for the large Stage C at the moment.
For Stage B the difference is not so big. The **large** Stage B is better at reconstructing small details,
but if your GPU is not so powerful, just go for the smaller one.

### Remark
Unfortunately, you can not run the models in float16 at the moment. Only bfloat16 or float32 work for now. However,
with some investigation, it should be possible to fix the overflowing and allow for inference in float16 as well.