import argparse
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description='Pedestrian Motion Lab')
    sub=parser.add_subparsers(dest='command',required=True)
    prepare=sub.add_parser('prepare');prepare.add_argument('--data',default='data')
    evaluate=sub.add_parser('evaluate');evaluate.add_argument('--data',default='data');evaluate.add_argument('--model',default='models/yolox_s.onnx');evaluate.add_argument('--output',default='reports/custom')
    serve=sub.add_parser('serve');serve.add_argument('--port',type=int,default=8786)
    args=parser.parse_args()
    if args.command=='prepare':
        from .data import prepare
        prepare(args.data)
    elif args.command=='evaluate':
        from .evaluate import evaluate
        evaluate(args.data,args.model,args.output)
    else:
        from .server import serve
        serve(args.port)

if __name__=='__main__':main()
