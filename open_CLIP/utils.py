import open_clip

OPENCLIP_MODEL_DIC = {
    'laion400m': {
        'ViT-B/32': ('laion400m_e32','ViT-B-32-quickgelu'),
        'ViT-B/16': ('laion400m_e32','ViT-B-16'),
        'ViT-L/14': ('laion400m_e32','ViT-L-14'),
    },
    'openai': {
        'ViT-B/32': ('openai','ViT-B-32-quickgelu'),
        'ViT-B/16': ('openai','ViT-B-16'),
        'ViT-L/14': ('openai','ViT-L-14')
    },
    'laion2b': {
        'ViT-B/32': ('laion2b_s34b_b79k','ViT-B-32'),
        'ViT-B/16': ('laion2b_s34b_b88k','ViT-B-16'),
        'ViT-L/14': ('laion2b_s32b_b82k','ViT-L-14')
    }
}


def get_engine(arch, mode='val', corpus='laion400m'):
    corpus_config ,model_arch = OPENCLIP_MODEL_DIC[corpus][arch]

    model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_arch, 
                                                                 pretrained=corpus_config)
        
    tokenizer = open_clip.get_tokenizer(model_arch)
    model = model.float() # Removes the mixed precision stuff. # 但是为什么

    return model, preprocess, tokenizer

