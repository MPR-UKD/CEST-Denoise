import subprocess


def remove_docker_image(image_name):
    # Remove containers using the image
    try:
        container_rm_command = [
            "docker",
            "rm",
            "-f",
            "$(docker ps -a -q --filter ancestor=" + image_name + ")",
        ]
        subprocess.run(container_rm_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print("No containers are using the image. Continuing to remove the image.")

    # Remove the image
    try:
        image_rm_command = ["docker", "rmi", "-f", image_name]
        subprocess.run(image_rm_command, check=True)
        print(f"Docker image {image_name} has been removed.")
    except subprocess.CalledProcessError as e:
        print("The image cannot be removed. It may be already removed or not exist.")


def build_and_push_docker_image(image_name, dockerfile_location):
    # Build the Docker image
    try:
        build_command = [
            "docker",
            "image",
            "build",
            "-t",
            image_name,
            dockerfile_location,
        ]
        subprocess.run(build_command, check=True)
        print(f"Docker image {image_name} has been created.")
    except subprocess.CalledProcessError as e:
        print("Failed to build the image.")

    # Push the image to Docker Hub
    try:
        push_command = ["docker", "image", "push", image_name]
        subprocess.run(push_command, check=True)
        print(f"Docker image {image_name} has been pushed to Docker Hub.")
    except subprocess.CalledProcessError as e:
        print("Failed to push the image to Docker Hub.")


# Use the functions
image_name = "lurad101/denoise"
dockerfile_location = "."

remove_docker_image(image_name)
build_and_push_docker_image(image_name, dockerfile_location)
