# 
You can find the overleaf repository at
https://www.overleaf.com/read/nthmmtjfvztc#d9e88b

The seminal paper for the main idea is https://arxiv.org/pdf/0911.5627v1

The confluence paper at https://confluence.cern.ch/pages/viewpage.action?pageId=539892459

The python code for the luminosity is at https://lhcmaskdoc.web.cern.ch/ipynbs/luminosity_formula/luminosity_formula/


And the newest https://github.com/ColasDroin/xtrack/blob/lumi_with_crabs/xtrack/lumi.py


## Main logic

The luminosity is in general a convolution in x, y, z. 
Let's assume factorization for the moment.

### Point 1
Can we integrate in Jx instead of x?
This would allow to have a simple way to see the effect of scraping w/o Abel transform.

### Point 2 
Removing the tail or the core at constant scarping.

### Point 3
Going to q-G distributions

### Point 4
Effect of misteering on the tail (simple approach)

### Point 5
Effect of the dipolar noise or diffusion models

