import argparse
import random
from itertools import cycle
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

MODEL = SentenceTransformer("all-MiniLM-L6-v2")  # 80 mb
# MODEL = SentenceTransformer("all-mpnet-base-v2")  # 420 mb
MIN_SIMILARITY = 0.9


@dataclass
class Flashcard:
    question: str = ""
    answer: str = ""
    correct_user_answers: list[str] = field(default_factory=list)
    wrong_user_answers: list[str] = field(default_factory=list)


def add_deck_to_flashcards(deck, flashcards):
    with open(deck, "r") as f:
        line = f.__iter__()
        flashcard = Flashcard()
        question = True
        while (current := next(line, None)) is not None:
            match current.strip():
                case "?":
                    question = False
                case "":
                    flashcards.append(flashcard)
                    flashcard = Flashcard()
                    question = True
                case _:
                    if question:
                        flashcard.question += current
                    else:
                        flashcard.answer += current

        flashcards.append(flashcard)


def is_correct(user_answer, flashcard):
    user_answer = MODEL.encode(user_answer)
    correct_answers = MODEL.encode([flashcard.answer] + flashcard.correct_user_answers)
    wrong_answers = MODEL.encode(flashcard.wrong_user_answers)

    def cos_sim(x, y):
        return dot(x, y) / (norm(x) * norm(y))

    sim_correct_anss = [cos_sim(user_answer, a) for a in correct_answers]
    sim_wrong_anss = [cos_sim(user_answer, a) for a in wrong_answers]

    sim_nearest_correct = max(sim_correct_anss)

    if len(sim_wrong_anss) > 0 and max(sim_wrong_anss) > sim_nearest_correct:
        return False
    elif sim_nearest_correct > MIN_SIMILARITY:
        return True
    else:
        return False


def main(decks, shuffle):
    flashcards = []
    for deck in decks:
        add_deck_to_flashcards(deck, flashcards)

    if shuffle:
        print("shuffling")
        random.shuffle(flashcards)

    for flashcard in cycle(flashcards):
        question = flashcard.question.strip()
        print("+" + "-" * len(question) + "+")
        print("|" + question + "|")
        print("+" + "-" * len(question) + "+")
        user_answer = input(">")
        correct = is_correct(user_answer, flashcard)
        print("+++++ CORRECT! :) +++++" if correct else "----- WRONG! :( -----")
        answer = flashcard.answer.strip()
        print("+" + "-" * len(answer) + "+")
        print("|" + answer + "|")
        print("+" + "-" * len(answer) + "+")
        assessment = "Was your answer correctly assessed? Y/n "
        while (user_input := input(assessment)) not in ["Y", "y", "N", "n", ""]:
            continue
        if user_input in ["N", "n"]:
            if correct:
                flashcard.wrong_user_answers.append(user_answer)
            else:
                flashcard.correct_user_answers.append(user_answer)
        print("\n~~~~~~~~~~~~~~~ Next Card ~~~~~~~~~~~~~~~\n")


if __name__ == "__main__":
    # python recall.py decks.txt --shuffle
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "decks",
        metavar="decks",
        type=str,
        nargs="+",
        help="the flashcards you want to recall",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="do you want to shuffle the decks?",
    )
    args = parser.parse_args()
    main(args.decks, args.shuffle)
