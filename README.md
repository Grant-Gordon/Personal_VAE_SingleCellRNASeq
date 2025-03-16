# Personal_VAE_SingleCellRNASeq
Gaining experience working with ML models and techniques 


Steps: 
1.	load data into sparse matrix 
2.	construct DAE 
		in ----> 512 --(relu)--> 128 --(relu)--> 512 --(relu)--> out 
		n = 0.001, adam, b=128
	plot loss in tensorboard

3.	VAE in pytorch 
4. 	dataloader to process all chunk
5.	Conditional layers
	GANS
	Multimodality
