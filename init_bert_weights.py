def init_bert_weights(self, module):
    """ Initialize the weights.
    """
    # if isinstance(module, (nn.Linear, nn.Embedding)):
    #     # Slightly different from the TF version which uses truncated_normal for initialization
    #     # cf https://github.com/pytorch/pytorch/pull/5617
    #     module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    if isinstance(module, nn.Linear):
	    # Slightly different from the TF version which uses truncated_normal for initialization
	    # cf https://github.com/pytorch/pytorch/pull/5617
	    module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
	elif isinstance(module, nn.Embedding):
	    # Slightly different from the TF version which uses truncated_normal for initialization
	    # cf https://github.com/pytorch/pytorch/pull/5617
	    if self.config.freeze_emb:
	    	module.weight.requires_grad = False
	    if self.config.init_emb_path:
	    	def load_word_embeddings(filepath):
		    	embeddings_index = {}
		    	with open(filepath, 'r') as f:
				    for line in f:
				        word, vec = line.split(' ', 1)
				        embeddings_index[word] = np.array(list(map(float, vec.split())))
		    	return embeddings_index

		    def load_vocab(filepath):
		    	vocab = []
		    	with open(filepath, 'r') as f:
				    for line in f:
				        word = line.trim()
				        vocab.append(word)
		    	return vocab

		    emb_size = model.weight.data.size(1)
		    emb_index = load_word_embeddings(self.config.init_emb_path)
		    dictionary = load_vocab(self.config.vocab_path)
	    	pretrained_weight = np.empty([len(dictionary), emb_size], dtype=float)
	    	for word in vocab::
	            if word in emb_index:
	                pretrained_weight[i] = emb_index[word]
	            else:
	                pretrained_weight[i] = np.random.normal(loc=0.0, 
	                										scale=std=self.config.initializer_range, 
	                										size=emb_size) # make sure you have 'import numpy as np'
	        module.weight.data.copy_(torch.from_numpy(pretrained_weight))
	    else:
	    	module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, BertLayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()