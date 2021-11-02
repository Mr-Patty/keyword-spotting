import onnx
import argparse
import torch
from model import SpeechResModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='models/model_v4.pt', help='Path to saved checkpoint')
    parser.add_argument('--output', default='models/model_v4.onnx', help='Path where to save onnx model')
    parser.add_argument('--labels_set', default=['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes'],
                        help='Set of labels')
    namespace = parser.parse_args()
    argv = vars(namespace)

    torch_checkpoint = argv['checkpoint']
    labels_set = argv['labels_set']
    output = argv['output']
    model = SpeechResModel(n_labels=len(labels_set)+2).float()

    model.load_state_dict(torch.load(torch_checkpoint))
    model.eval()
    model.cpu()

    batch_size = 1
    x = torch.randn(batch_size, 101, 40, requires_grad=False)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,  # model being run
                      x,  # model input (or a tuple for multiple inputs)
                      output,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                    'output': {0: 'batch_size'}})

    onnx_model = onnx.load(output)
    onnx.checker.check_model(onnx_model)
