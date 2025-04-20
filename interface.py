"""
This module isolates all direct communication with external Large Language 
Model APIs. It provides the mechanisms for generating text responses while 
incorporating behavioral steering based on an agent's internal feature state, 
performing contrastive analyses to identify features differentiating between 
interaction sets, and requesting external judgments on game outcomes. It also 
handles low-level details like API key management, request/response formatting, 
and error handling for these external service interactions.
"""
