// Sokoban solver: push-optimal per room, then stitched through the hallway.
//
// Idea: every level is a 1-wide hallway (the player's row) plus 1-4 rooms, each
// room joined to the hallway by exactly one door cell. A box pushed out of a room
// can never come back, so rooms are independent: the level's push-optimal count
// is the sum of the rooms' optima. We solve each room alone with A* over pushes
// (not steps), then walk the player between rooms along the hallway.
//
// Per-room search:
//   - state = sorted box cells + the player's reachable region (canonicalised to
//     its smallest cell), so walking never creates new states;
//   - cost = pushes; heuristic = minimum assignment of boxes to goals using
//     shortest-path (wall-aware) distances, which is admissible;
//   - dead squares: cells from which a box can never reach a goal (reverse push
//     reachability, ignoring other boxes) are pruned, both as box positions and
//     as push targets;
//   - a solution is accepted only if the player can still walk out to the hallway
//     afterwards, so rooms stay stitchable;
//   - budgets are counted in expansions, never in wall-clock time, so two runs
//     are byte-identical. If A* exhausts its budget, a greedy best-first search
//     (same moves, heuristic only) tries to find any solution instead.
//
// No clock, no randomness, no environment reads: fully deterministic.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

const DIRS: [number, number, string][] = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

type Level = {
  w: number;
  h: number;
  wall: boolean[];
  goals: number[];
  goalSet: Set<number>;
  boxes: number[];
  player: number;
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall: boolean[] = new Array(w * h).fill(true);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = false;
      if ('.*+'.includes(ch)) goals.push(i);
      if ('$*'.includes(ch)) boxes.push(i);
      if ('@+'.includes(ch)) player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet: new Set(goals), boxes, player };
}

/** BFS distances from `from` over `floor` cells (walls excluded). */
function bfsDist(w: number, floor: Set<number>, from: number): Map<number, number> {
  const dist = new Map<number, number>([[from, 0]]);
  const q = [from];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const d = dist.get(c)!;
    const cx = c % w;
    const cy = (c - cx) / w;
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0) continue;
      const n = ny * w + nx;
      if (!floor.has(n) || dist.has(n)) continue;
      dist.set(n, d + 1);
      q.push(n);
    }
  }
  return dist;
}

/** Shortest walk path (as move letters) from `from` to `to` over `floor`, with
 *  `blocked` cells treated as walls. Empty string if unreachable or equal. */
function walkPath(
  w: number,
  floor: Set<number>,
  blocked: Set<number>,
  from: number,
  to: number,
): string | null {
  if (from === to) return '';
  const prev = new Map<number, [number, string]>();
  const q = [from];
  prev.set(from, [-1, '']);
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w;
    const cy = (c - cx) / w;
    for (const [dx, dy, letter] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0) continue;
      const n = ny * w + nx;
      if (!floor.has(n) || blocked.has(n) || prev.has(n)) continue;
      prev.set(n, [c, letter]);
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) {
          const [p, l] = prev.get(cur)!;
          s = l + s;
          cur = p;
        }
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

/** Cells reachable from `from` over `floor`, with `blocked` as walls. */
function reachable(
  w: number,
  floor: Set<number>,
  blocked: Set<number>,
  from: number,
): Set<number> {
  const seen = new Set<number>([from]);
  const q = [from];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w;
    const cy = (c - cx) / w;
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0) continue;
      const n = ny * w + nx;
      if (!floor.has(n) || blocked.has(n) || seen.has(n)) continue;
      seen.add(n);
      q.push(n);
    }
  }
  return seen;
}

type PushStep = { walk: string; push: string };

/**
 * Push search over one region.
 * `floor`: cells the player and boxes may occupy (room cells + entry cell).
 * `boxes0`, `goals`: sorted box cells and goal cells inside the region.
 * `startPlayer`: where the player begins (the hallway cell by the door, or the
 * level's player for a whole-level search).
 * `exitCell`: cell the player must be able to reach after solving (the hallway
 * cell by the door), or -1 when no exit is needed.
 * Returns the push sequence (walk + push per step) or null.
 */
function solveRegion(
  w: number,
  floor: Set<number>,
  boxes0: number[],
  goals: number[],
  startPlayer: number,
  exitCell: number,
  budget: number,
  greedy: boolean,
): PushStep[] | null {
  const goalSet = new Set(goals);
  const n = boxes0.length;
  // Pairwise shortest-path distances from every floor cell to every goal.
  const distToGoal: Map<number, number>[] = goals.map((g) => bfsDist(w, floor, g));
  const cellDist = (cell: number, gi: number): number => {
    const d = distToGoal[gi]!.get(cell);
    return d === undefined ? 1e9 : d;
  };
  // Dead squares: reverse push reachability from the goals, ignoring other
  // boxes. A cell is live if it is a goal, or some push from it lands on a
  // live cell while the player can stand on the opposite side.
  const live = new Set<number>(goals);
  let changed = true;
  while (changed) {
    changed = false;
    for (const c of floor) {
      if (live.has(c)) continue;
      const cx = c % w;
      const cy = (c - cx) / w;
      for (const [dx, dy] of DIRS) {
        const tx = cx + dx;
        const ty = cy + dy;
        const sx = cx - dx;
        const sy = cy - dy;
        if (tx < 0 || ty < 0 || sx < 0 || sy < 0) continue;
        const t = ty * w + tx;
        const s = sy * w + sx;
        if (live.has(t) && floor.has(s)) {
          live.add(c);
          changed = true;
          break;
        }
      }
    }
  }
  const dead = new Set<number>();
  for (const c of floor) if (!live.has(c)) dead.add(c);

  // Minimum assignment of boxes to goals (n <= 4, so permutations are cheap).
  const heuristic = (boxes: number[]): number => {
    if (n === 0) return 0;
    const idx = goals.map((_, i) => i);
    let best = 1e9;
    const permute = (arr: number[], k: number): void => {
      if (k === arr.length) {
        let s = 0;
        for (let i = 0; i < n; i++) {
          s += cellDist(boxes[i]!, arr[i]!);
          if (s >= best) return;
        }
        if (s < best) best = s;
        return;
      }
      for (let i = k; i < arr.length; i++) {
        [arr[k], arr[i]] = [arr[i]!, arr[k]!];
        permute(arr, k + 1);
        [arr[k], arr[i]] = [arr[i]!, arr[k]!];
      }
    };
    permute(idx, 0);
    return best;
  };

  const solved = (boxes: number[]): boolean => boxes.every((b) => goalSet.has(b));
  const canExit = (boxes: number[], player: number): boolean => {
    if (exitCell < 0) return true;
    const blocked = new Set(boxes);
    return reachable(w, floor, blocked, player).has(exitCell);
  };

  // A* (or greedy best-first) over push states.
  const sBoxes: number[][] = [boxes0.slice()];
  const sPlayer: number[] = [startPlayer];
  const sG: number[] = [0];
  const sParent: number[] = [-1];
  const sStep: (PushStep | null)[] = [null];
  const keyOf = (boxes: number[], player: number): string => {
    const blocked = new Set(boxes);
    const region = reachable(w, floor, blocked, player);
    let rep = 1e9;
    for (const c of region) if (c < rep) rep = c;
    return `${boxes.join(',')}|${rep}`;
  };
  const best = new Map<string, number>([[keyOf(boxes0, startPlayer), 0]]);
  // Heap entries: [priority, h, seq, stateIndex]. Greedy: priority = h.
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const h0 = heuristic(boxes0);
  heap.push([greedy ? h0 : h0, h0, seq++, 0]);
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] !== b[0] ? a[0] < b[0] : a[1] !== b[1] ? a[1] < b[1] : a[2] < b[2];
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      [heap[i], heap[p]] = [heap[p]!, heap[i]!];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        [heap[i], heap[c]] = [heap[c]!, heap[i]!];
        i = c;
      }
    }
    return top;
  };

  let expanded = 0;
  while (heap.length > 0 && expanded < budget) {
    const [, , , si] = heapPop();
    if (best.get(keyOf(sBoxes[si]!, sPlayer[si]!)) !== si) continue;
    const boxes = sBoxes[si]!;
    const player = sPlayer[si]!;
    if (solved(boxes)) {
      if (canExit(boxes, player)) {
        const steps: PushStep[] = [];
        for (let t = si; t > 0; t = sParent[t]!) steps.unshift(sStep[t]!);
        return steps;
      }
      continue;
    }
    expanded++;
    const blocked = new Set(boxes);
    const region = reachable(w, floor, blocked, player);
    for (let bi = 0; bi < n; bi++) {
      const b = boxes[bi]!;
      const bx = b % w;
      const by = (b - bx) / w;
      for (const [dx, dy, letter] of DIRS) {
        const sx = bx - dx;
        const sy = by - dy;
        const tx = bx + dx;
        const ty = by + dy;
        if (sx < 0 || sy < 0 || tx < 0 || ty < 0) continue;
        const stand = sy * w + sx;
        const target = ty * w + tx;
        if (!region.has(stand)) continue;
        if (!floor.has(target) || blocked.has(target)) continue;
        if (dead.has(target) && !goalSet.has(target)) continue;
        const nb = boxes.slice();
        nb[bi] = target;
        nb.sort((a, b2) => a - b2);
        if (nb.some((c) => dead.has(c) && !goalSet.has(c))) continue;
        const walk = walkPath(w, floor, blocked, player, stand);
        if (walk === null) continue;
        const g = sG[si]! + 1;
        const k = keyOf(nb, b);
        const seen = best.get(k);
        if (seen !== undefined && sG[seen]! <= g) continue;
        const t = sBoxes.length;
        sBoxes.push(nb);
        sPlayer.push(b);
        sG.push(g);
        sParent.push(si);
        sStep.push({ walk, push: letter });
        best.set(k, t);
        const h = heuristic(nb);
        heapPush([greedy ? h : g + h, h, seq++, t]);
      }
    }
  }
  return null;
}

/** Solve one level, returning the full move string (or '' if unsolved). */
function solveLevel(lv: Level): string {
  const { w, wall, goals, boxes, player } = lv;
  const py = (player - (player % w)) / w;
  // Floor cells of the whole level.
  const allFloor = new Set<number>();
  for (let i = 0; i < wall.length; i++) if (!wall[i]) allFloor.add(i);
  // Hallway row cells.
  const hallway = new Set<number>();
  for (let x = 0; x < w; x++) {
    const c = py * w + x;
    if (!wall[c]) hallway.add(c);
  }
  // Rooms: connected components of non-wall cells outside the hallway row.
  const seen = new Set<number>();
  type Room = { cells: Set<number>; boxes: number[]; goals: number[]; door: number; entry: number };
  const rooms: Room[] = [];
  let decompositionOk = true;
  for (const c of allFloor) {
    const cy = (c - (c % w)) / w;
    if (cy === py || seen.has(c)) continue;
    const cells = new Set<number>();
    const q = [c];
    seen.add(c);
    for (let qi = 0; qi < q.length; qi++) {
      const cur = q[qi]!;
      cells.add(cur);
      const cx = cur % w;
      const cyy = (cur - cx) / w;
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx;
        const ny = cyy + dy;
        if (nx < 0 || ny < 0 || ny >= lv.h) continue;
        const n2 = ny * w + nx;
        if (ny === py || wall[n2] || seen.has(n2)) continue;
        seen.add(n2);
        q.push(n2);
      }
    }
    const rBoxes = boxes.filter((b) => cells.has(b));
    const rGoals = goals.filter((g) => cells.has(g));
    const doors = [...cells].filter((cell) => {
      const cx = cell % w;
      const cyy = (cell - cx) / w;
      return Math.abs(cyy - py) === 1 && hallway.has(cyy < py ? (py * w + cx) : (py * w + cx));
    });
    if (doors.length !== 1 || rBoxes.length !== rGoals.length || rBoxes.length === 0) {
      decompositionOk = false;
      break;
    }
    const door = doors[0]!;
    rooms.push({ cells, boxes: rBoxes, goals: rGoals, door, entry: py * w + (door % w) });
  }
  // Every box must belong to exactly one room for the decomposition to hold.
  if (decompositionOk) {
    const inRoom = new Set<number>();
    for (const r of rooms) for (const b of r.boxes) inRoom.add(b);
    if (inRoom.size !== boxes.length) decompositionOk = false;
  }

  if (!decompositionOk) {
    // Fallback: push search over the whole level, no exit requirement.
    const steps = solveRegion(w, allFloor, boxes, goals, player, -1, 400_000, false)
      ?? solveRegion(w, allFloor, boxes, goals, player, -1, 400_000, true);
    if (steps === null) return '';
    let moves = '';
    let cur = player;
    let curBoxes = boxes.slice();
    for (const st of steps) {
      moves += st.walk;
      for (const ch of st.walk) {
        const [dx, dy] = DIRS.find((d) => d[2] === ch)!;
        cur = cur + dy * w + dx;
      }
      const [dx, dy] = DIRS.find((d) => d[2] === st.push)!;
      // The player steps onto the box cell and pushes that box one cell on.
      const boxCell = cur + dy * w + dx;
      const idx = curBoxes.indexOf(boxCell);
      if (idx < 0) return '';
      curBoxes[idx] = boxCell + dy * w + dx;
      curBoxes.sort((a, b) => a - b);
      moves += st.push;
      cur = boxCell;
    }
    return moves;
  }

  // Solve rooms nearest-first along the hallway.
  const remaining = rooms.slice();
  let curPlayer = player;
  let moves = '';
  while (remaining.length > 0) {
    let bestIdx = 0;
    let bestDist = 1e9;
    for (let i = 0; i < remaining.length; i++) {
      const d = Math.abs(remaining[i]!.entry - curPlayer);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    const room = remaining.splice(bestIdx, 1)[0]!;
    // Walk along the hallway to this room's entry cell.
    const hallWalk = walkPath(w, hallway, new Set<number>(), curPlayer, room.entry);
    if (hallWalk === null) return '';
    moves += hallWalk;
    curPlayer = room.entry;
    // Solve the room. Its floor is the room's cells plus the entry cell.
    const floor = new Set<number>(room.cells);
    floor.add(room.entry);
    const steps =
      solveRegion(w, floor, room.boxes, room.goals, room.entry, room.entry, 400_000, false)
      ?? solveRegion(w, floor, room.boxes, room.goals, room.entry, room.entry, 400_000, true);
    if (steps === null) return '';
    // Replay the room's pushes, tracking boxes and the player.
    let curBoxes = room.boxes.slice();
    for (const st of steps) {
      moves += st.walk;
      for (const ch of st.walk) {
        const [dx, dy] = DIRS.find((d) => d[2] === ch)!;
        curPlayer = curPlayer + dy * w + dx;
      }
      const [dx, dy] = DIRS.find((d) => d[2] === st.push)!;
      const boxCell = curPlayer + dy * w + dx;
      const idx = curBoxes.indexOf(boxCell);
      if (idx < 0) return '';
      curBoxes[idx] = boxCell + dy * w + dx;
      curBoxes.sort((a, b) => a - b);
      moves += st.push;
      curPlayer = boxCell;
    }
    // Walk out of the room to the hallway entry cell.
    const exitWalk = walkPath(w, floor, new Set(curBoxes), curPlayer, room.entry);
    if (exitWalk === null) return '';
    moves += exitWalk;
    curPlayer = room.entry;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8')));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
