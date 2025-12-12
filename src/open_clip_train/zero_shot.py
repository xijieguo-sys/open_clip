import logging

import torch
from tqdm import tqdm

from open_clip import get_input_dtype, get_tokenizer, build_zero_shot_classifier, \
    IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, IMAGENET_A_CLASSNAMES, IMAGENET_R_CLASSNAMES, CIFAR10_CLASSNAMES
from open_clip_train.precision import get_autocast
import os
import json


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def run(model, classifier, dataloader, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    with torch.inference_mode():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5


def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data and 'imagenet-a' not in data and 'imagenet-r' not in data and 'cifar10-c' not in data and 'imagenet-c' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}
    if args.distributed and not args.horovod:
        model = model.module

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)

    if 'imagenet-val' in data or 'imagenet-v2' in data:
        with autocast():
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
            )

        logging.info('Using classifier')
        results = {}
        if 'imagenet-val' in data:
            top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
            results['imagenet-zeroshot-val-top1'] = top1
            results['imagenet-zeroshot-val-top5'] = top5
        if 'imagenet-v2' in data:
            top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
            results['imagenetv2-zeroshot-val-top1'] = top1
            results['imagenetv2-zeroshot-val-top5'] = top5

    if 'imagenet-a' in data:
        logging.info('Building zero-shot classifier for ImageNet-A')
        with autocast():
            classifier_a = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_A_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
            )
        logging.info('Using ImageNet-A classifier')
        results = {}
        top1, top5 = run(model, classifier_a, data['imagenet-a'].dataloader, args)
        results['imagenet-a-zeroshot-val-top1'] = top1
        results['imagenet-a-zeroshot-val-top5'] = top5

    if 'imagenet-r' in data:
        logging.info('Building zero-shot classifier for ImageNet-R')
        with autocast():
            classifier_a = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_R_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
            )
        logging.info('Using ImageNet-R classifier')
        results = {}
        top1, top5 = run(model, classifier_a, data['imagenet-r'].dataloader, args)
        results['imagenet-r-zeroshot-val-top1'] = top1
        results['imagenet-r-zeroshot-val-top5'] = top5

    if 'cifar10-c' in data:
        corruption = getattr(args, 'cifar10_c_corruption', 'gaussian_noise')
        severity = getattr(args, 'cifar10_c_severity', 'all')
    
        logging.info(f'Building zero-shot classifier for CIFAR-10-C ({corruption}, severity={severity})')

        with autocast():
            classifier_a = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=CIFAR10_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
            )
        logging.info('Using Cifar10-C classifier')
        results = {}
        top1, top5 = run(model, classifier_a, data['cifar10-c'].dataloader, args)
        results['cifar10-c-zeroshot-val-top1'] = top1
        results['cifar10-c-zeroshot-val-top5'] = top5

    if 'imagenet-c' in data:
        corruption = getattr(args, 'imagenet_c_corruption', 'gaussian_noise')
        severity = getattr(args, 'imagenet_c_severity', 1)
        
        logging.info(f'Building zero-shot classifier for ImageNet-C ({corruption}, severity={severity})')

        with autocast():
            classifier = build_zero_shot_classifier(
                model,
                tokenizer=tokenizer,
                classnames=IMAGENET_CLASSNAMES,
                templates=OPENAI_IMAGENET_TEMPLATES,
                num_classes_per_batch=10,
                device=device,
                use_tqdm=True,
            )
        
        logging.info(f'Using ImageNet-C classifier')
        results = {}
        top1, top5 = run(model, classifier, data['imagenet-c'].dataloader, args)
        results[f'imagenet-c-{corruption}-s{severity}-zeroshot-top1'] = top1
        results[f'imagenet-c-{corruption}-s{severity}-zeroshot-top5'] = top5

        results_dir = os.path.join(args.logs, 'imagenet_c_results')
        os.makedirs(results_dir, exist_ok=True)

        pretrained_name = os.path.basename(args.pretrained).replace('.pt', '') if args.pretrained else 'none'
        results_file = os.path.join(results_dir, f'{pretrained_name}_{corruption}_s{severity}.json')

        results_data = {
            'corruption': corruption,
            'severity': severity,
            'model': args.model,
            'pretrained': args.pretrained,
            'top1': float(top1),
            'top5': float(top5)
        }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

    logging.info('Finished zero-shot imagenet.')

    return results
