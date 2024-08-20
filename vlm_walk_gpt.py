# encoding: utf-8

import os

import argparse
from tqdm import tqdm, trange

from utils.io_utils import load_access_street_view, load_task_data, init_seed, \
    get_graph_and_images_dual, shrink_graph, parse_json
from utils.io_utils import ask_start_image, ask_middle_image, ask_summary_image


def output_words(city, index, text, model_dir, model_name):
    file_name = f"{model_dir}/supervised/{city}/{model_name}/{index}.md"
    os.makedirs(f"{model_dir}/supervised/{city}/{model_name}/", exist_ok=True)

    # 打开文件，追加内容，然后关闭文件
    with open(file_name, 'a') as file:
        file.write("\n")
        file.write(text)
        file.write("\n")


def output_images(city, index, image, model_dir, model_name):
    file_name = f"{model_dir}/supervised/{city}/{model_name}/{index}_image.md"
    os.makedirs(f"{model_dir}/supervised/{city}/{model_name}/", exist_ok=True)

    # 打开文件，追加内容，然后关闭文件
    with open(file_name, 'a') as file:
        file.write("\n")
        file.write(str(image))
        file.write("\n")


def get_image_cnt(city, index, path, model_dir, model_name):
    path = f"{model_dir}/supervised/{city}/{model_name}/{index}_image.md"

    if not os.path.exists(path):
        return False

    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()

    images = [i for i in list(content.split('\n')) if i]

    return len(images) >= 10


def random_walk(g, start_point, args, index, llm, tokenizer, images, city, model_dir):
    output_words(city, index, "# Experimental Result", model_dir, llm)
    output_words(city, index, "##  Start Edge Caption", model_dir, llm)

    id = int(g.nodes[start_point]["image"])
    image = images[id]
    output_words(city, index, f"![no_name](../../data/{city}/{index}/squeeze_images/{id}.jpg)", model_dir, llm)
    output_images(city, index, id, model_dir, llm)

    images_emb = []
    summary = ask_start_image(llm, tokenizer, image, "", args)
    output_words(city, index, summary, model_dir, llm)

    for i in trange(9):
        try:
            neighbors = list(g.neighbors(start_point))
            best_neighbor = -1
            best_answer = -1
            best_image = -1
            for neighbor in neighbors:
                id = int(g.nodes[neighbor]["image"])
                now_image = images[id]
                answer, reason = ask_middle_image(llm, tokenizer, now_image, summary, args)

                output_words(city, index,
                             f"![no_name](../../data/{city}/{index}/squeeze_images/{id}.jpg)", model_dir, llm)
                output_words(city, index, reason, model_dir, llm)

                if answer > best_answer:
                    best_answer = answer
                    best_neighbor = neighbor
                    best_image = now_image
                    output_words(city, index, "best answer updated!!!!!!!!!!!!!!!!", model_dir, llm)

            start_point = best_neighbor

            output_words(city, index, f"###  update summary", model_dir, llm)
            summary = ask_summary_image(llm, tokenizer, best_image, summary, args)
            output_words(city, index, summary, model_dir, llm)

            output_images(city, index, int(g.nodes[best_neighbor]["image"]), model_dir, llm)
        except Exception as e:
            print(e)
            continue


def evaluate(loader, args, llm, tokenizer, city, model_dir, image_dir):
    for index in tqdm(loader):
        index = int(index)

        if get_image_cnt(city, index, "", model_dir, llm):
            print(f"skip {index}")
            continue
        else:
            if os.path.exists(f"{model_dir}/supervised/{city}/{llm}/{index}_image.md"):
                os.remove(f"{model_dir}/supervised/{city}/{llm}/{index}_image.md")
                os.remove(f"{model_dir}/supervised/{city}/{llm}/{index}.md")

        sub_g, street_views, images = get_graph_and_images_dual(index, args.city_size, image_dir)

        new_g, start_point = shrink_graph(sub_g)

        random_walk(new_g, start_point, args, index, llm, tokenizer, images, city, model_dir)


city_names = ["New York City", "San Francisco", "Washington", "Chicago"]


def main(args):
    # pip install timm==0.3.2
    # todo: change timm to 1.0.7
    print(args.llm)

    image_dir, model_dir, save_dir, log_dir, task_dir = parse_json(args)

    # init_seed(args.seed)

    city = args.city_size

    llm, tokenizer = args.llm, 0

    test_dataset = [args.test_data]

    evaluate(test_dataset, args, llm, tokenizer, city_names[city], model_dir, image_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--location",
        type=str,
        default="local",
    )

    parser.add_argument(
        "--save_name",
        type=str,
        default="test",
        help="save_name",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="ViT",
        choices=["MAE", "ResNet", "SimCLR", "CLIP", "ViT"],
        help="model name",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="batch_size",
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=100000,
        help="num epochs",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default="cuda:0",
        help="device",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="lr",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="patience",
    )

    parser.add_argument(
        "--city_size",
        type=int,
        default=0,
        help="number of cities",
    )

    parser.add_argument(
        "--target",
        type=int,
        default=0,
        help="Carbon or Population",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed",
    )

    parser.add_argument(
        "--test_data",
        type=str,
        default="1306",
        help="test_grid",
    )

    parser.add_argument(
        "--llm",
        type=str,
        default="Claude",
        choices=["minicpm", "FireLLaVA", "Gemini", "Claude"],
        help="llm",
    )

    args = parser.parse_args()

    main(args)
