
from skimage import measure
from detectron2.layers.mask_ops import paste_masks_in_image
from detectron2.structures.masks import PolygonMasks


def swap_x_y_in_countour(contour):
    assert len(contour) % 2 == 0, f"len:{len(contour)}"
    for i in range(0,len(contour), 2):
        contour[i], contour[i+1] = contour[i+1], contour[i]
    return contour

def countour_to_list_and_optional_extend(countour):
    flatten_countur = countour.ravel().tolist()

    if len(flatten_countur) < 6:
        flatten_countur = flatten_countur + flatten_countur[:2] # TODO could be better

    flatten_countur = swap_x_y_in_countour(flatten_countur)
    
    return flatten_countur

def binary_mask_to_countour(mask):
    contours = measure.find_contours(mask.cpu().numpy(), 0.5)
    return [countour_to_list_and_optional_extend(countur) for countur in contours]



def get_polygon_masks_from_predictions(prediction_masks, prediction_boxes, image_shape, mask_thres=0.5):
        pseudo_masks = paste_masks_in_image(prediction_masks[:, 0, :, :], prediction_boxes, image_shape)

        all_countours = []
            
        for mask in pseudo_masks:
            all_countours.append(binary_mask_to_countour(mask))
        return PolygonMasks(all_countours)
    