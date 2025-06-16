##
import importlib

##
def load_model(opt, dataloader,classes):
    """ Load model based on the model name.

    Arguments:
        opt {[argparse.Namespace]} -- options
        dataloader {[dict]} -- dataloader class

    Returns:
        [model] -- Returned model
    """
    model_name = opt.model
    model_path = f"lib.models.{model_name}"
    model_lib  = importlib.import_module(model_path)
    
    # Handle special naming for video model
    if model_name == 'ocr_gan_video':
        model_class_name = 'Ocr_Gan_Video'
    else:
        model_class_name = model_name.title()
    
    model = getattr(model_lib, model_class_name)
    return model(opt, dataloader,classes)