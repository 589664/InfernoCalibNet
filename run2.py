from src.XrayDataset import XRayDataset


def main():
    dataset = XRayDataset()

    # Get the first data sample
    image, labels = dataset[0]
    print(image.shape, labels)


if __name__ == "__main__":
    main()
