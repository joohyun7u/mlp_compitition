import utils.tool_box as T


if __name__ == '__main__':
    clean_image_dir = '/local_datasets/MLinP/train/clean/'
    noise_output_dir = '/local_datasets/MLinP/train/MSN/'

    noiser = T.Noiser(
        clean_image_dir = clean_image_dir,
        noise_output_dir = noise_output_dir
    )

    noiser.make_some_noise()