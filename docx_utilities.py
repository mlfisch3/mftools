import os
import docx
from docx import Document

#  SaveImages(fname):


def SaveImages(fname):
	doc = Document(fname)

	rels = {}
	for r in doc.part.rels.values():
		if isinstance(r._target, docx.parts.image.ImagePart):
			rels[r.rId] = os.path.basename(r._target.partname)

	for rid in list(rels.keys()):
		bytes_of_image = doc.inline_shapes.part.related_parts[rid].image.blob
		with open(rels[rid], 'wb') as f:
			f.write(bytes_of_image)
			f.close()
