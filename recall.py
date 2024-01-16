import argparse
import random
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer
from numpy import dot
from numpy.linalg import norm

MODEL = SentenceTransformer("all-MiniLM-L6-v2")


@dataclass
class Flashcard:
    question: str = ""
    answer: str = ""
    correct_answers: list[str] = field(default_factory=list)
    wrong_answers: list[str] = field(default_factory=list)


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
    user_answer, original_answer = MODEL.encode([user_answer, flashcard.answer])
    correct_answers = MODEL.encode(flashcard.correct_answers)
    wrong_answers = MODEL.encode(flashcard.wrong_answers)

    def cos_sim(x, y):
        return dot(x, y) / norm(x) * norm(y)

    sim_orig_ans = cos_sim(user_answer, original_answer)
    sim_correct_anss = [cos_sim(user_answer, a) for a in correct_answers]
    sim_wrong_anss = [cos_sim(user_answer, a) for a in wrong_answers]

    sim_correct = (
        sim_orig_ans
        if len(sim_correct_anss) == 0
        else max(sim_orig_ans, *sim_correct_anss)
    )
    sim_wrong = 0 if len(sim_wrong_anss) == 0 else max(sim_wrong_anss)

    if sim_wrong > sim_correct:
        return False
    elif sim_correct > 0.9:
        return True
    else:
        return False


def main(decks):
    flashcards = []
    for deck in decks:
        add_deck_to_flashcards(deck, flashcards)

    # random.shuffle(flashcards)

    while True:
        flashcard = random.choice(flashcards)
        print(flashcard.question)
        print("Type your answer:")
        user_answer = input()
        correct = is_correct(user_answer, flashcard)
        print("CORRECT! :)" if correct else "WRONG! :(")
        print("Answer:", flashcard.answer)
        print("Was you answer correctly assessed? Y/n")
        while (user_input := input()) not in ["Y", "y", "N", "n", ""]:
            print("Please type Y or n")
        if user_input in ["N", "n"]:
            if correct:
                flashcard.wrong_answers.append(user_answer)
            else:
                flashcard.correct_answers.append(user_answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "decks",
        metavar="decks",
        type=str,
        nargs="+",
        help="the flashcards you want to recall",
    )
    args = parser.parse_args()
    main(args.decks)
