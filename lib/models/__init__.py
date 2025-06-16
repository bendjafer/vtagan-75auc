import importlib

def load_model(opt, dataloader, classes):
    """Load model based on the model name."""
    model_name = opt.model
    model_path = f"lib.models.{model_name}"
    model_lib = importlib.import_module(model_path)
    
    if model_name == 'gan_model':
        model_class_name = 'Gan_Model'
    else:
        model_class_name = model_name.title()
    
    model = getattr(model_lib, model_class_name)
    return model(opt, dataloader, classes)