from full_code import *


def go_baby_go():
    # --- braa --- preprocessing ,transformation and augmentation.
    train_transform, val_transform = transformation()

    # --- mohammed --- load and split Dataset
    train_loader, val_loader, full_dataset = loading(train_transform, val_transform)

    # --- allawiii --- load pretrained efficientNet and define loss function
    model = load_pretrained_architecture()
    criterion, optimizer, scheduler, scaler = loss(model)

    # --- Ezz --- train and visualize
    train_losses, train_accuracies = train_model(
        model, criterion, optimizer, scheduler, scaler, train_loader
    )
    visualize_result(train_losses, train_accuracies)

    # --- mahmood --- evaluation
    evaluate_model(model, full_dataset, val_loader)

    torch.save(model.state_dict(), "./cat_dog_war.pth")
