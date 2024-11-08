mkdir models
cd models
# Maybe go to .../temp/.../models and symlink
wget https://github.com/ggjy/DeLVM/releases/download/hf-ckpt/llama_300m_hf.zip 
unzip llama_300m_hf.zip
rm llama_300m_hf.zip
# From https://drive.google.com/drive/folders/1CyucT_QOArUH_Au8dfzRSwseyiCGserF?usp=share_link
gdown --folder 1CyucT_QOArUH_Au8dfzRSwseyiCGserF
