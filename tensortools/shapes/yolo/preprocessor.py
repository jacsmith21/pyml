from clients.eigen.shapes.preprocessor import ShapePreprocessor
from common.utils import get_frame_object, extract_roi


class Preprocessor(ShapePreprocessor):
    def extract_outputs(self, event):
        eig = get_frame_object(self.config.scratch_path, event['S3Key'])
        eig.convert_to_iron()
        image = eig.imgiron

        rois = []
        labels = []
        for validation in event['validations']:
            roi = extract_roi(validation)
            rois.extend(roi)

            id_norm = self.config.outputs['labels'][validation['labelId']]
            labels.extend(id_norm)

        for roi, label in zip(rois, labels):
            pass
