'''

Interactive cmd line demo of multi-tagged output, input with multi-tags and process iteratively

'''

import os
import pdb
import torch
import nltk
import random
import argparse
import logging
import numpy as np
from tqdm import tqdm
from termcolor import colored
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import (
	GPT2LMHeadModel,
	GPT2Tokenizer
)
from signal import signal, SIGINT
from sys import exit
import warnings
random.seed(42)

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected.')
    exit(0)

def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

torch.manual_seed(42)
os.environ["TOKENIZERS_PARALLELISM"] = "true"


MAX_LENGTH = 1000
TAG_SET = ['biography', 'concession', 'idiom', 'description', 'sub', 'cause', 'effect']
ANS_TOK = '<answer>'
SEP_TOK = '<sep>'
EOS_TOK = '<|endoftext|>'

def set_seed(args):
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	if args.n_gpu > 0:
		torch.cuda.manual_seed_all(args.seed)

def adjust_length_to_model(length, max_sequence_length):
	if length < 0 and max_sequence_length > 0:
		length = max_sequence_length
	elif 0 < max_sequence_length < length:
		length = max_sequence_length  # No generation bigger than model size
	elif length < 0:
		length = MAX_LENGTH  # avoid infinite loop
	return length

def process_prompt(t):

	assert any([f'<{tg}>' in t for tg in TAG_SET]), "Need to have writing tags"
	t = sent_tokenize(t)
	# assert len(t) >= 2, "expecting previous context as a complete sentence"
	# assert not any([tg in t[0] for tg in TAG_SET]), "expecting tag in the second sentence"

	for i in range(len(t)):
		t[i] = ' '.join(word_tokenize(t[i]))

	for tag in TAG_SET:
		for i in range(len(t)):
			t[i] = t[i].replace(f'< {tag} >', f'<{tag}>')

	t = ' '.join(t) + ' ' + SEP_TOK
	t = t.replace('>,', '> ,').replace('>.', '> .')
	return t

def post_process(prompt, decoded, tag):

	decoded = decoded.split(ANS_TOK)
	prompt_sp = prompt.split()

	sep_id = prompt_sp.index(SEP_TOK)
	prompt_text = ' '.join(prompt_sp[:sep_id])
	prompt_text = prompt_text.split(f'<{tag}>')
	# all_text = [p[0].lstrip().rstrip() + " " + p[1].lstrip().rstrip() for p in zip(prompt_text, decoded)]
	all_text = [colored(p[0].lstrip().rstrip(), "red") + " " + colored(p[1].lstrip().rstrip(), "green") for p in zip(prompt_text, decoded)]
	all_text_nocolor = [p[0].lstrip().rstrip()+ " " + p[1].lstrip().rstrip() for p in zip(prompt_text, decoded)]
	
	all_text = ' '.join(all_text).replace(SEP_TOK, "").replace(EOS_TOK, "")
	all_text_nocolor = ' '.join(all_text_nocolor).replace(SEP_TOK, "").replace(EOS_TOK, "")
	return all_text, all_text_nocolor

def generate(prompt_text, tokenizer, model, args, output=False, tag='biography', idx=-1):

	enc_prompt = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]

	enc_prompt = enc_prompt.to(args.device)

	if enc_prompt.size()[-1] == 0:
		input_ids = None
	else:
		input_ids = enc_prompt

	out_seqs = model.generate(
		input_ids=input_ids,
		max_length=args.length + len(enc_prompt[0]),
		temperature=args.t,
		top_k=args.k,
		top_p=args.p,
		repetition_penalty=1.,
		do_sample=True,
		num_return_sequences=args.n,
		pad_token_id=tokenizer.pad_token_id
		)

	if len(out_seqs.shape) > 2:
		out_seqs.squeeze_()
	gen_seqs, gen_seqs_nocol = [], []
	for gen_seq_idx, gen_seq in enumerate(out_seqs):

		gen_seq = gen_seq.tolist()
		text = tokenizer.decode(gen_seq, clean_up_tokenization_spaces=True)
		text = text[: text.find(args.stop_token) if args.stop_token else None]
		decoded = (text[len(tokenizer.decode(enc_prompt[0], clean_up_tokenization_spaces=True)) :])

		full_sequence, full_sequence_nocol = post_process(prompt_text, decoded, tag)

		gen_seqs.append(full_sequence)
		gen_seqs_nocol.append(full_sequence_nocol)
		
		if output:
			if idx > 0:
				print(f"DECODED {idx}: {full_sequence}\n")
			else:
				print(f"DECODED {gen_seq_idx}: {full_sequence}\n")

	return gen_seqs, gen_seqs_nocol

def extract_tag_specific_context(prompt):
	'''
		e.g. w1 w2 <t1> w3 w4 <t2> w5 <t3> w6 w7 <t3> <sep>
		return 
			[
				(<t1>, 'w1 w2 <t1> w3 w4'), ('<t2>', '<t2> w5'), ('<t3>', '<t3> w6 w7 <t3> ')
			]
	'''
	ret = []
	last_tag = ''
	this_tag_context = []
	for w in prompt.split():
		if w.startswith('<') and w.endswith('>'):
			if last_tag == '' or w == last_tag:
				this_tag_context.append(w)
			else:
				ret.append((last_tag, ' '.join(this_tag_context)))
				this_tag_context = [w]
			last_tag = w
		else:
			this_tag_context.append(w)
	return ret

def generate_rephrase(prompt_text, tokenizer, model, args):
	enc_prompt = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt", padding=True)["input_ids"]

	enc_prompt = enc_prompt.to(args.device)

	if enc_prompt.size()[-1] == 0:
		input_ids = None
	else:
		input_ids = enc_prompt

	out_seqs = model.generate(
		input_ids=input_ids,
		max_length=args.length + len(enc_prompt[0]),
		temperature=args.t,
		top_k=args.k,
		top_p=args.p,
		repetition_penalty=1.,
		do_sample=True,
		num_return_sequences=args.n,
		pad_token_id=tokenizer.pad_token_id
		)

	if len(out_seqs.shape) > 2:
		out_seqs.squeeze_()

	gen_seqs, gen_seqs_nocol = [], []
	for gen_seq_idx, gen_seq in enumerate(out_seqs):

		gen_seq = gen_seq.tolist()
		text = tokenizer.decode(gen_seq, clean_up_tokenization_spaces=True)
		text = text[: text.find(args.stop_token) if args.stop_token else None]
		decoded = (text[len(tokenizer.decode(enc_prompt[0], clean_up_tokenization_spaces=True)) :])
		decoded = decoded.replace(EOS_TOK, '').replace(ANS_TOK, '')
		decoded = ' '.join(word_tokenize(decoded))
		gen_seqs.append(decoded)
	return gen_seqs
		

def process_substitution_tag(prompt_text, model, tokenizer, args):

	sp = prompt_text.split('<sub>')

	if len(sp) == 1: # no <sub> tag
		return [prompt_text] * args.n

	assert len(sp) == 3, 'incorrect usage of <sub> tag'
	#assert len(sp) % 2 == 1, 'invalid usage of <sub> tag'

	ret = []
	for i, span in enumerate(sp):
		if i % 2 == 0:
			continue

		ssp = span.split()
		assert not any(w.startswith('<') and w.endswith('>') for w in ssp), \
				'you cannot embed tags within sub tag'

		prompt = '<sub> ' + span + ' <sub> <sep>'
		rephrased = generate_rephrase(prompt, tokenizer, model, args)
		for r in rephrased:
			this_sp = sp.copy()
			this_sp[i] = r
			ret.append(' '.join(this_sp))

	return ret

def color(prompt, p):
	'''find spans between the tags and change the term color'''
	spp = prompt.split()
	spp = ['<tag>' if w.startswith('<') and w.endswith('>') and w != '<sub>' else w for w in spp]

	if '<sub>' in spp:
		spp = ' '.join(spp).split('<sub>')
		spp[1] = '<tag> <tag>'
	spp = ' '.join(spp).split('<tag>')
	for span in spp:
		s = span.lstrip().rstrip()
		if s != '.' and len(s) > 0:
			p = p.replace(s, f" ||| {s} ||| ")
	p = p.split('|||')
	
	p = [colored(p[i].lstrip().rstrip(), "green") if i % 2 == 0 else \
			colored(p[i].lstrip().rstrip(), "red") for i in range(len(p))]
	return ' '.join(p)


def generate_mix(prompt_text, model, tokenizer, args):
	'''process different tags iteratively'''
	prompt_ = process_prompt(prompt_text)
	
	# before inserting, check for any substitution tag
	prompt = process_substitution_tag(prompt_, model, tokenizer, args)
	
	for i, pt in enumerate(prompt):

		args.n = 1
		prompt_tag_specific_span = extract_tag_specific_context(pt)
		if len(prompt_tag_specific_span) == 1:
			this_tag = prompt_tag_specific_span[0][0].lstrip('<').rstrip('>')
			generate(pt, tokenizer, model, args, output=True, tag=this_tag, idx=i)


		elif len(prompt_tag_specific_span) == 0:
			pt = pt.replace(SEP_TOK, "")
			pt = colored(pt, "green")
			print(f"DECODED {i} : {pt}\n")

		else:
			prompts = [prompt_tag_specific_span[0][1] + ' ' + SEP_TOK] * args.n
			args.n = 1
			col_span = []
			for si, span in enumerate(prompt_tag_specific_span):

				this_gen = []
				last_tag = span[0].lstrip('<').rstrip('>')
				for p in prompts:
					full_seq, full_seq_nocol = generate(p, tokenizer, model, args, tag=last_tag)
					this_gen.append(full_seq_nocol[0])

				if si < len(prompt_tag_specific_span) - 1:
					prompts = [p + ' ' + prompt_tag_specific_span[si+1][1] + ' ' + SEP_TOK  for p in this_gen]

			for pi, p in enumerate(this_gen):
				print(f"DECODED {i} : {color(prompt_, p)}\n")

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--k", type=int, default=30)
	parser.add_argument("--p", type=float, default=0.9)
	parser.add_argument("--n", type=int, default=3)
	parser.add_argument("--prompt", type=str, default='')
	parser.add_argument("--model-path", type=str, default=None)
	parser.add_argument("--length", type=int, default=100)
	parser.add_argument("--gpu-id", type=int, default=3)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--stop-token", type=str, default=EOS_TOK)
	parser.add_argument("--t",type=float,default=1.0,help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
	)
	args = parser.parse_args()

	args.device = torch.device(f"cuda:{args.gpu_id}")
	args.n_gpu = 1
	set_seed(args)

	model_class, tokenizer_class = GPT2LMHeadModel, GPT2Tokenizer

	tokenizer = tokenizer_class.from_pretrained(args.model_path)
	model = model_class.from_pretrained(args.model_path)
	model.to(args.device)

	signal(SIGINT, handler)
	while True:
		try:
			old_n = args.n
			input_prompt = "Type input with tags, tags allowed: <concession> <description> <idiom> <biography> <sub> <cause> <effect> \n"
			prompt_text = args.prompt if args.prompt != '' else \
							input(input_prompt)
			generate_mix(prompt_text, model, tokenizer, args)
			args.prompt = ''
			args.n = old_n
		except:
			print("do you want to exit? [yes|no]\n")
			response = input()
			if response == "yes":
				exit(0)
			else:
				continue

if __name__ == "__main__":

	main()