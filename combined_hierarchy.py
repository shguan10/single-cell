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

	"UBERON:0002048 lung":9,
	"UBERON:0000115 lung epithelium":9,

	"UBERON:0000955 brain":10,
	"UBERON:0002038 substantia nigra":10,
	"UBERON:0001891 midbrain":10,

	"UBERON:0001954 Ammon's horn":11,
	"UBERON:0001898 hypothalamus":11,

	"UBERON:0002435 striatum":12,
	"UBERON:0010743 meningeal cluster":12,


	"UBERON:0001003 skin epidermis":13,
	"UBERON:0001997 olfactory epithelium":13,
	"UBERON:0001902 epithelium of small intestine":13,

	"UBERON:0000473 testis":14,
	"UBERON:0000992 female gonad":14,
	"UBERON:0000922 embryo":14,

	"UBERON:0001264 pancreas":15,
	"UBERON:0000007 pituitary gland":15,

	"UBERON:0000045 ganglion":16,
	"UBERON:0000044 dorsal root ganglion":16,
	"UBERON:0004129 growth plate cartilage":16,

	"UBERON:0000966 retina":17,
	"UBERON:0002107 liver":17,
	"UBERON:0001851 cortex":17
}

cell_superclasses = ['connective tissue', 'other', 'endocrine',
    'embryo', 'muscle', 'leukocytes', 'haem/{stem, leuk}', 'brain',
    'haem stem', 'lung', 'brain-main', 'brain-2', 'brain-3',
    'epithelium', 'reproduction', 'endocrine', 'bone', 'other']

bad_phrase = "ABCDEFGHIJ"

stage_1_filename = "combined_stage_1_models.pickle"
stage_2_filename = "combined_stage_2_models.pickle"



