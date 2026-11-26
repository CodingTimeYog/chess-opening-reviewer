"""Mistake detection and feedback: facts, deterministic template text, trigger events."""

from chess_trainer.feedback.facts import FeedbackBuilder, classify_loss
from chess_trainer.feedback.template import render_event, render_move
from chess_trainer.feedback.triggers import Event, TriggerEngine

__all__ = [
    "Event",
    "FeedbackBuilder",
    "TriggerEngine",
    "classify_loss",
    "render_event",
    "render_move",
]
