import argparse
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from model.autoencoder import AutoEncoder
from model.vq_autoencoder import VQAutoEncoder
from classifier import AudioClassifier
from utils import get_dataloaders
from tqdm import tqdm

def setup_trainer(args, model_name):
    """ Set up PyTorch Lightning trainer with checkpointing """
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints',
        filename=f'{model_name}-{args.epochs:02d}-{args.lr:.2f}',
        save_top_k=1,
        mode='min'
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=1 if torch.cuda.is_available() and args.gpu else 0,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True,
        enable_progress_bar=True
    )
    return trainer

def train_model(args, model, model_name):
    """ General function to handle model training """
    train_loader, validation_loader = get_dataloaders()
    trainer = setup_trainer(args, model_name)
    trainer.fit(model, train_loader, validation_loader)
    torch.save(model.state_dict(), args.save_path)

def evaluate_model(args, model, dataloader, model_name):
    """ Function to evaluate the model """
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    
    for x, y in tqdm(dataloader):
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        predictions = torch.argmax(y_pred, dim=1)
        correct += (predictions == y).sum().item()
        total += y.size(0)
    
    accuracy = 100 * correct / total
    print(f"{model_name} Accuracy: {accuracy:.2f}%")
    return accuracy

def recon_accuracy(args):
    """ Function to evaluate model on reconstructed data """
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    _, valid_dataloader = get_dataloaders(normalize=True)
    classifier = AudioClassifier(lr=args.lr)
    classifier.load_state_dict(torch.load(args.load_clf, map_location=device))
    classifier = classifier.to(device)

    vqautoencoder = VQAutoEncoder(learning_rate=args.lr)
    vqautoencoder.load_state_dict(torch.load(args.load_vqvae, map_location=device))
    vqautoencoder = vqautoencoder.to(device).eval()

    correct, total = 0, 0
    
    for x, y in tqdm(valid_dataloader):
        x, y = x.to(device), y.to(device)
        recon, _ = vqautoencoder(x)
        y_pred = classifier(recon)
        predictions = torch.argmax(y_pred, dim=1)
        correct += (predictions == y).sum().item()
        total += y.size(0)
    
    accuracy = 100 * correct / total
    print(f"Reconstruction Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models")
    parser.add_argument("--train_ae", action="store_true", help="Train AutoEncoder")
    parser.add_argument("--train_vqae", action="store_true", help="Train VQ-AutoEncoder")
    parser.add_argument("--train_clf", action="store_true", help="Train Classifier")
    parser.add_argument("--eval_clf", action="store_true", help="Evaluate the Classifier")
    parser.add_argument("--recon_acc", action="store_true", help="Evaluate model on reconstructed data")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--save_path", type=str, default="model.pth", help="Path to save the trained model")
    parser.add_argument("--load_clf", type=str, default="classifier.pth", help="Path to load the classifier for evaluation")
    parser.add_argument("--load_vqvae", type=str, default="vqvae.pth", help="Path to load the VQ-AutoEncoder for evaluation")
    args = parser.parse_args()

    if args.train_ae:
        model = AutoEncoder(learning_rate=args.lr)
        train_model(args, model, "AutoEncoder")
    elif args.train_vqae:
        model = VQAutoEncoder(learning_rate=args.lr)
        train_model(args, model, "VQAutoEncoder")
    elif args.train_clf:
        model = AudioClassifier(lr=args.lr)
        train_model(args, model, "Classifier")
    elif args.eval_clf:
        model = AudioClassifier(lr=args.lr)
        model.load_state_dict(torch.load(args.load_clf))
        _, valid_dataloader = get_dataloaders(normalize=True)
        evaluate_model(args, model, valid_dataloader, "Classifier")
    elif args.recon_acc:
        recon_accuracy(args)

if __name__ == '__main__':
    main()
