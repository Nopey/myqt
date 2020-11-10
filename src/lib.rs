use generational_arena::{Arena,Index};

/// A coordinate in 2D space
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Coord(u32, u32);

impl Coord {
    pub(crate) fn to_qcoord(&self, magnitude: u8) -> QCoord {
        QCoord(
            Self::half_qcoord(self.0, magnitude),
            Self::half_qcoord(self.1, magnitude)
        )
    }
    fn half_qcoord(u: u32, m: u8) -> u32 {
        u.reverse_bits() >> (32 - m as u32)
    }
    pub fn max(&self) -> u32 {
        u32::max(self.0, self.1)
    }
    pub fn neighbors(&self) -> () {
        todo!()
    }
}

impl std::ops::Add for Coord {
    type Output = Coord;
    #[inline]
    fn add(self, rhs: Coord) -> Coord {
        Coord(self.0 + rhs.0, self.1 + rhs.1)
    }
}

/// A coordinate in a quadtree.
///
/// This has reverse bit order as Coord, with the 1's position being
///    the root quad tree's offset, 2's pos being the root's child, etc
#[derive(Copy, Clone, Debug)]
pub(crate) struct QCoord(u32, u32);
impl QCoord {
    pub fn pop_childid(&mut self) -> ChildId {
        let ret = ChildId(
             (self.0 & 1)==1,
             (self.1 & 1)==1,
        );
        self.0 >>= 1;
        self.1 >>= 1;
        ret
    }
}

/// One of the four child
#[derive(Copy, Clone, Debug)]
struct ChildId(bool, bool);
impl ChildId {
    fn to_index(self) -> usize {
        match self {
            ChildId(false, false) => 0,
            ChildId( true, false) => 1,
            ChildId(false,  true) => 2,
            ChildId( true,  true) => 3,
        }
    }
}

/// Stores sparse T's in 2D space with integer coordinates
#[derive(Clone, Debug)]
pub struct QuadTree <T> {
    items: Arena<T>,
    nodes: Arena<Node>,
    root: Index,
    /// The # of layers of internal nodes
    depth: u8,
}

impl<T> QuadTree<T> {
    pub fn new() -> Self {
        let items = Arena::new();
        let mut nodes = Arena::new();
        let root = nodes.insert(Node::default());
        QuadTree {
            items,
            nodes,
            root,
            depth: 1
        }
    }
    /// First coord is a position, and the Second is a postiive offset
    /// from the position, forming a rectangle.
    pub fn query<F>(&self, filter: F) -> impl Iterator<Item=(Coord, &T)>
        where F: FnMut(Coord, Coord) -> bool
    {
        Query::new(self, filter)
    }
    pub fn query_mut<F>(&mut self, filter: F) -> impl Iterator<Item=(Coord, &mut T)>
        where F: FnMut(Coord, Coord) -> bool
    {
        QueryMut::new(self, filter)
    }

    fn width(&self) -> u32 {
        1 << (self.depth as u32)
    }
    fn contains(&self, coord: Coord) -> bool {
        coord.max() < self.width()
    }
    /// Ensure that `coord` is within the quadtrees space, by adding more
    /// root nodes.
    /// Currently private, as I can't think of an external use for this.
    fn reserve(&mut self, coord: Coord) {
        let new_depth = 32-u32::leading_zeros(coord.max()) as u8;
        if new_depth <= self.depth {
            return;
        }
        // Empty root node has no leaves, so can be freely scaled
        if self.nodes[self.root].children.iter().any(Option::is_some) {
            // We don't have an empty root node, so we must add parent
            // nodes to the root to compensate for the scaling.
            for _ in self.depth..new_depth {
                let new_root = self.nodes.insert(Default::default());
                self.nodes[ new_root].children[0] = Some(self.root);
                self.nodes[self.root].parent = Some(new_root);
                self.root = new_root;
            }
        }
        self.depth = new_depth;
    }

    pub fn get(&self, coord: Coord) -> Option<&T> {
        // Discard get requests outside of the quadtree
        if !self.contains(coord) {
            return None;
        }

        // TODO: Move this loop into a lookup function
        let mut qcoord = coord.to_qcoord(self.depth);
        let mut node = self.root;
        for _ in 0..self.depth {
            let childid = qcoord.pop_childid();
            node = self.nodes[node].children[childid.to_index()]?;
        }

        Some(&self.items[node])
    }

    pub fn get_mut(&mut self, coord: Coord) -> Option<&mut T> {
        // Discard get requests outside of the quadtree
        if !self.contains(coord) {
            return None;
        }

        let mut qcoord = coord.to_qcoord(self.depth);
        let mut node = self.root;
        for _ in 0..self.depth {
            let childid = qcoord.pop_childid();
            node = self.nodes[node].children[childid.to_index()]?;
        }

        Some(&mut self.items[node])
    }
    pub fn set(&mut self, coord: Coord, item: T) -> Option<T> {
        // Grow to meet set requests outside of the quadtree
        self.reserve(coord);

        let mut qcoord = coord.to_qcoord(self.depth);
        let mut node = self.root;
        for depth in 0..self.depth {
            let childidx = qcoord.pop_childid().to_index();
            node = if let Some(node) = self.nodes[node].children[childidx]{
                node
            } else {
                if depth+1 == self.depth {
                     self.nodes[node].children[childidx] = Some(self.items.insert(item));
                     return None;
                }
                let child = self.nodes.insert(Default::default());
                self.nodes[child].parent = Some(node);
                self.nodes[node].children[childidx] = Some(child);
                child
            }
        }
        Some(std::mem::replace(&mut self.items[node], item))
    }
    /// TODO: If performance is of concern, rewrite the `remove` function
    /// This could also allow it to collapse empty nodes (all child ren are None)
    /// And collapse NOP root nodes (root node with one child on the Top Left)
    pub fn remove(&mut self, coord: Coord) -> Option<T> {
        self.map(coord, |_| None)
    }
    pub fn map<F: FnOnce(Option<T>) -> Option<T> >(&mut self, _coord: Coord, _func: F) -> Option<T> {
        todo!()
    }
}

impl<T> Default for QuadTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

//TODO: Exported reference type, referencing a leaf.

/// An internal node
#[derive(Clone, Debug, Default)]
struct Node {
    parent: Option<Index>,
    children: [Option<Index>; 4],
}

/// Doesn't own the collection, so can't be a std::Iterator.
/// Can't own the collection, so we can implement query_mut.
struct QueryInner<F: FnMut(Coord, Coord)->bool> {
    trace: Box<[(u8, Coord)]>,
    current: Index,
    filter: F,
}

impl<F: FnMut(Coord, Coord)->bool> QueryInner<F> {
    fn new<T>(qt: &QuadTree<T>, filter: F) -> Self {
        QueryInner {
            trace: vec![(0, Coord(0,0)); usize::max(1, qt.depth as usize)].into_boxed_slice(),
            current: qt.root,
            filter,
        }
    }

    fn next<T>(&mut self, qt: &QuadTree<T>) -> Option<(Coord, Index)> {
        let mut depth = if self.trace[qt.depth as usize - 1].0 == 0 {
            qt.depth - 1
        } else {
            0
        };
        loop {
            // We are currently `depth` layers from the items.
            let (trace, pos) = self.trace[depth as usize];
            if trace < 4 {
                // Go down, or try to.
                self.trace[depth as usize].0 += 1;
                let width = 1 << depth;
                // half width
                let offset = match trace {
                    0 => Coord(    0,     0),
                    1 => Coord(width,     0),
                    2 => Coord(    0, width),
                    3 => Coord(width, width),
                    _ => unreachable!()
                };
                if (self.filter)(pos, Coord(width, width)) {
                    if let Some(ndepth) = depth.checked_sub(1) {
                        if let Some(child) = qt.nodes[self.current].children[trace as usize] {
                            self.current = child;
                            self.trace[ndepth as usize].0 = 0;
                            self.trace[ndepth as usize].1 = pos + offset;
                            depth = ndepth;
                        }
                    } else if let Some(index) = qt.nodes[self.current].children[trace as usize] {
                        return Some((pos + offset, index));
                    }
                }
            } else {
                // Go up
                self.current = qt.nodes[self.current].parent?;
                depth += 1;
            }
        }
    }
}

struct Query<'a, T, F: FnMut(Coord, Coord)->bool> {
    query_inner: QueryInner<F>,
    quad_tree: &'a QuadTree<T>,
}
impl<'a, T, F: FnMut(Coord, Coord)->bool> Query<'a, T, F> {
    fn new(quad_tree: &'a QuadTree<T>, filter: F) -> Self {
        Query {
            query_inner: QueryInner::new(quad_tree, filter),
            quad_tree,
        }
    }
}
impl<'a, T, F: FnMut(Coord, Coord)->bool> Iterator for Query<'a, T, F> {
    type Item = (Coord, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        self.query_inner.next(self.quad_tree).map(|(pos, index)| (pos, &self.quad_tree.items[index]))
    }
}

struct QueryMut<'a, T: 'a, F: FnMut(Coord, Coord)->bool> {
    query_inner: QueryInner<F>,
    quad_tree: &'a mut QuadTree<T>,
}
impl<'a, T: 'a, F: FnMut(Coord, Coord)->bool> QueryMut<'a, T, F> {
    fn new(quad_tree: &'a mut QuadTree<T>, filter: F) -> Self {
        QueryMut {
            query_inner: QueryInner::new(quad_tree, filter),
            quad_tree,
        }
    }
}
impl<'a, T: 'a, F: FnMut(Coord, Coord)->bool> Iterator for QueryMut<'a, T, F> {
    type Item = (Coord, &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        self.query_inner.next(self.quad_tree).map(|(pos, index)|
            // SAFETY: The QueryInner semiiterator does not yield the same item twice.
            (pos, unsafe { std::mem::transmute ( &mut self.quad_tree.items[index] ) } )
        )
    }
}

#[cfg(test)]
mod test {
    use crate::{Coord, QuadTree};
    #[test]
    fn quadtree_basic_query() {
        let mut qt = QuadTree::new();
        const COUNT: u32 = 100;
        for i in 0..COUNT {
            let x = i * 101 % 199;
            let y = i * 107 % 197;
            qt.set(Coord(x, y), Coord(x, y));
        }
        assert_eq!(qt.query(|_, _| true ).map(|(a, &b)| assert_eq!(a, b) ).count() as u32, COUNT);
    }
}
