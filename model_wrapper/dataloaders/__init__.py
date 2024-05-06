from dataloaders.datasets import atria, heart, heartStructure, pericardium, ventricles
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):
    print(args.dataset)
    if args.dataset == 'atria':
        test_set = atria.AtriaSegmentation(args, split='test')
        num_class = test_set.NUM_CLASSES
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        return test_loader, num_class
    if args.dataset == 'heartStructure':
        test_set = heartStructure.HeartStructureSegmentation(args, split='test')
        num_class = test_set.NUM_CLASSES
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        return test_loader, num_class
    if args.dataset == 'heart':
        test_set = heart.HeartSegmentation(args, split='test')
        num_class = test_set.NUM_CLASSES
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        return test_loader, num_class
    if args.dataset == 'pericardium':
        test_set = pericardium.PericardiumSegmentation(args, split='test')
        num_class = test_set.NUM_CLASSES
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        return test_loader, num_class
    if args.dataset == 'ventricles':
        test_set = ventricles.VentriclesSegmentation(args, split='test')
        num_class = test_set.NUM_CLASSES
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)
        return test_loader, num_class
    else:
        raise NotImplementedError

