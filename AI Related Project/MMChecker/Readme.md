
## Section 1: AI Agent Configuration
- **Primary AI Algorithm:** Minimax with Alpha-Beta Pruning
- **Pruning Technique:**  Alpha-Beta Pruning
- **Evaluation Heuristic Strategy:**  Piece Advantage, Positional Weights, Weighting on the King

---

## Section 2: Complexity & Evaluation Function Analysis
- **Board Configuration:** 8x8 board, 64element list with 2D render, piece sit only in the dark square, the white are on 0 to 2 and the black are on rows 5 to 7 and they move horizontally,
- **Piece Valuation & Rules:** There are four piece black, white, king black, king white. The get_available_move  capture of opponent and their move while make move did the move of the piece. The undo move support to the minmax search to look ahead.
- **Heuristic Features Used:** Calculate how many pieces and kings each side has. More weight on king as it could freely move diagonally on the board. (my_pieces - opp_pieces) + 3x(my_kings - opp_kings)
 + 0.1 x (my_chance - opp_chance)

---

## Section 3: Deployed and Presentation Information
- **Deployment Platform:**  Render
- **Live Deployment URL:** https://checker-mmdt002.onrender.com
