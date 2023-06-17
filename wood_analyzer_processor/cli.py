import click
import wood_analyzer_processor.common as ia


@click.group()
def cli():
    pass


@click.command()
@click.option("--path", "-p", required=True, help="Base path to the image directory")
def generate_predictions(path: str):
    ia.predict_images(path)


@click.command()
@click.option("--path", "-p", required=True, help="Base path to the image directory")
def postprocess_predictions(path: str):
    ia.postprocess_images(path)


@click.command()
@click.option("--path", "-p", required=True, help="Base path to the image directory")
def generate_segmentations(path: str):
    ia.generate_segmentations(path)


@click.command()
@click.option("--path", "-p", required=True, help="Base path to the image directory")
@click.option("--filename", "-f", required=True, help="Filename of the image")
def generate_segmentation(path: str, filename: str):
    ia.generate_segmentation(path, filename)


@click.command()
@click.option("--path", "-p", required=True, help="Base path to the image directory")
@click.option("--filename", "-f", required=True, help="Filename of the segmentation")
def calculate_measurements(path: str, filename: str):
    ia.calculate_measurements(path, filename)


cli.add_command(generate_predictions)
cli.add_command(postprocess_predictions)
cli.add_command(generate_segmentations)
cli.add_command(generate_segmentation)
cli.add_command(calculate_measurements)

if __name__ == "__main__":
    cli()
