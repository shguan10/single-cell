superclass_dict = {

	'CL:0000057 fibroblast': 0,
	'CL:0000137 osteocyte': 0,

	'CL:1000497 kidney cell': 1,
	'CL:0008019 mesenchymal cell': 1,
	'CL:0002365 medullary thymic epithelial cell' : 1,

	'CL:0000163 endocrine cell': 2,
	'CL:0000169 type B pancreatic cell': 2,

	'CL:0002321 embryonic cell': 3,
	'CL:0000353 blastoderm cell': 3,
	'CL:0002322 embryonic stem cell': 3,

	'CL:0000192 smooth muscle cell': 4,
	'CL:0000746 cardiac muscle cell': 4,

	'CL:0001056 dendritic cell, human': 5,
	'CL:0000084 T cell': 5,
	'CL:0000235 macrophage': 5,

	'CL:0000081 blood cell': 6,
	'CL:0000763 myeloid cell': 6,

	'CL:0002319 neural cell': 7,
	'CL:0000540 neuron': 7,
	'CL:0000127 astrocyte': 7,

	'CL:0002034 long term hematopoietic stem cell': 8,
	'CL:0002033 short term hematopoietic stem cell': 8,
	'CL:0000037 hematopoietic stem cell': 8,
}

cell_superclasses = ['connective tissue', 'other', 'endocrine',
    'embryo', 'muscle', 'leukocytes', 'haem/{stem, leuk}', 'brain',
    'haem stem']

bad_phrase = "UBERON:"

stage_1_filename = "cl_stage_1_models.pickle"
stage_2_filename = "cl_stage_2_models.pickle"




