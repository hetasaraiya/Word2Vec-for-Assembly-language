import logging
import importlib
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from gensim.models.word2vec import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec

reload(logging)
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')

import gensim

class GetSentences(object):
	def __init__(self, dirname):
			self.dirname = dirname
	def __iter__(self):
	#def GetSentences(dirname):
		l=0
		vector_reg=['v512','v256','v128','mm','xmm','ymm','zmm']
		sentences=[]
		#for dir2name in os.listdir(self.dirname):
		with open('models/comp1',"r") as f:
			for line in f:
				
				dir2name=line.split("\n")[0]
				#if not(dir2name.startswith("grep")):
				#	continue
				for fname in os.listdir(os.path.join(self.dirname, dir2name)):
					for line in open(os.path.join(self.dirname,dir2name, fname)):
					#yield line.split()
						sentence = line.split()
						new_sentence = []
						
						for word in sentence:
							parts = word.split('_')
							temp=parts[0]
							if temp.startswith("rex"):
								continue
							for i in range(1,len(parts)):
								w=parts[i]
								if w.lower().startswith(tuple(vector_reg)):
									if parts[0].lower().startswith("p"):
										temp="packed"
									else:
										temp="vector"
									break
							new_sentence.append(temp)
					
						#sentences.append(new_sentence)
						yield new_sentence
					#l=l+1
		#print "There are"+ str(l)+" sentences"
#		return sentences

def checking(dirname):
	file=open('vector.txt','w')
	vector_reg=['v512','v256','v128','mm','xmm','ymm','zmm']
	flag=False
	for dir2name in os.listdir(dirname):
		for fname in os.listdir(os.path.join(dirname, dir2name)):
			for line in open(os.path.join(dirname, dir2name, fname)):
				
				sentence = line.split()
				new_sentence = []
				
				for word in sentence:
					parts = word.split('_')
					temp=parts[0]
					for i in range(1,len(parts)):
						w=parts[i]
						if w.lower().startswith(tuple(vector_reg)):
							if parts[0].lower().startswith("p"):
								temp="package"
							else:
								temp="vector"
							break
					if temp.startswith(("fs_")):
						print(dir2name)
						print(fname)
						print(word)
						flag=True
					if flag:
						break
				if flag:
					break
			if flag:
				break
		if flag:
			break
					

class EpochSaver(CallbackAny2Vec):
	def __init__(self,path_prefix):
		self.path_prefix=path_prefix
		self.epoch=0
		self.loss=0
	def on_epoch_end(self,model):
		output_path=self.path_prefix+'_epoch_'+str(self.epoch)+'.out'
		model.wv.save_word2vec_format('models/models_epoch/binary_'+output_path,binary=True)
		print("Epoch_"+str(self.epoch))
		print("Training loss:"+str(model.get_latest_training_loss()))
		print("Diff Training loss:"+str(model.get_latest_training_loss()-self.loss))
		self.loss=model.get_latest_training_loss()
		self.epoch+=1
sentences = GetSentences('/filer/corpus')
epoch_saver=EpochSaver("model_some7")
print("Window=9 alpha=0.0005 sg min_count=3 size=45 iter=20 models_some7")
model = Word2Vec(sentences, min_count=3, sg=1, size=45, window=9,iter=20,alpha=0.0005, workers=4,compute_loss=True,callbacks=[epoch_saver])
model.wv.save_word2vec_format('models/model_some7_binary.out',binary=True)
model.save('models/model_some7.out')
