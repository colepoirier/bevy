use bevy_core::prelude::*;
use bevy_ecs::prelude::*;
use bevy_transform::prelude::*;
//use serde::{Deserialize, Serialize};
use private::*;
use smallvec::{smallvec, SmallVec};

// TODO: Create a simplified version without the children ?!

/// Provides a way of describing a hierarchy or named entities
/// and means for finding then in the world
///
/// By default and to save memory the nodes are indexed using `u16`
/// but you can change it when needed.
#[derive(Debug, Clone)]
pub struct Hierarchy<I: Index = u16> {
    /// Entity identification made by parent index and name
    entities: Vec<(I, Name)>,
    // ? NOTE: SmallVec<[u16; 10]> occupy the same 32 bytes as the SmallVec<[u16; 8]>, but the latter
    // ? should be only take 24 bytes using the "union" feature
    children: Vec<SmallVec<[I; 10]>>,
}

impl<I: Index> Default for Hierarchy<I> {
    fn default() -> Self {
        Self {
            // ? NOTE: Since the root has no parent in this context it points to a place outside the vec bounds
            entities: vec![(I::MAX_VALUE, Name::default())],
            children: vec![smallvec![]],
        }
    }
}

impl<I: Index> Hierarchy<I> {
    /// Used when the hierarchy must be in a specific order,
    /// this function takes an vec of entities defined by their parent index
    /// (on the same vec) and name.
    ///
    /// Any root entity should be indexed using `I::MAX_VALUE` or `I::MAX`.
    ///
    /// Many different root nodes are supported although having other roots
    /// make hard to search entities, please refer to the documentation of
    /// `find_entity` or `find_entity_in_world` to see how.
    pub fn from_ordered_entities(entities: Vec<(I, Name)>) -> Self {
        let mut children = vec![];
        children.resize_with(entities.len(), || smallvec![]);

        for (entity_index, (parent_index, _)) in entities.iter().enumerate() {
            children
                .get_mut(parent_index.as_usize())
                .map(|c| c.push(I::from_usize_checked(entity_index)));
        }

        Self { entities, children }
    }

    /// Number of entities registered.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Iterates over each entity parent index, name and children indexes
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = (&(I, Name), &[I])> {
        self.entities
            .iter()
            .zip(self.children.iter().map(|c| &c[..]))
    }

    /// Gets the entity parent index and `Name` components
    #[inline(always)]
    pub fn get_entity(&self, entity_index: I) -> &(I, Name) {
        &self.entities[entity_index.as_usize()]
    }

    pub fn depth_first<F: FnMut(I, &Name)>(&self, entity_index: I, visitor: &mut F) {
        let i = entity_index.as_usize();
        let (_, name) = &self.entities[i];

        visitor(entity_index, name);

        for child_index in &self.children[i] {
            self.depth_first(*child_index, visitor);
        }
    }

    /// Adds a new entity hierarchy path separated by backslashes (`'/'`)
    /// return the entity index and if was or not inserted
    pub fn get_or_insert_entity(&mut self, entity_path: &str) -> (I, bool) {
        let mut entity_created = false;
        let mut entity = I::from_usize(0); // Start search from root
        for name in entity_path.split('/') {
            // Ignore the first '/' or '///'
            if name.is_empty() {
                continue;
            }

            if let Some(e) = self
                .entities
                .iter()
                .position(|(p, n)| (*p, n.as_str()) == (entity, name))
            {
                // Found entity
                // ? NOTE: Conversion will never panic because the collection
                // ? size will only increase in the else branch where a
                // ? safe cast is performed
                entity = I::from_usize(e);
            } else {
                // Add entity
                let e = self.entities.len();
                self.entities.push((entity, Name::from_str(name)));
                self.children.push(smallvec![]);
                entity_created = true;
                // Soft limit added to save memory, identical to the curve limit
                let _parent = entity;
                entity = I::from_usize_checked(e);
                self.children[_parent.as_usize()].push(entity)
            }
        }

        (entity, entity_created)
    }

    /// Returns the entity path if found.
    ///
    /// The `NamedHierarchy` stores a the entity path in a specific way to improve search performance
    /// thus it needs to rebuilt in the human readable format
    pub fn get_entity_path_at(&self, mut entity_index: I) -> Option<String> {
        let mut path = None;

        while let Some((parent_index, name)) = self.entities.get(entity_index.as_usize()) {
            if let Some(path) = path.as_mut() {
                *path = format!("{}/{}", name.as_str(), path);
            } else {
                path = Some(name.as_str().to_string());
            }

            entity_index = *parent_index;
        }

        path
    }

    /// Finds an entity given a set of queries, see the example bellow
    /// how to proper call this function,
    ///
    /// ```rust
    /// let mut entities_table_cache = vec![];
    /// entities_table_cache.resize(clip.hierarchy.len(), None);
    /// // Assign the root entity as the first element
    /// entities_table_cache[0] = Some(root);
    ///
    /// let found_entity = named_hierarchy.find_entity(2, &mut entities_table_cache, children_query, name_query);
    /// ```
    ///
    /// *NOTE* Keep in mind that you can have as many root as you want
    /// but each root must be manually find and inserted in the `entities_table_cache`
    /// before calling this function.
    pub fn find_entity(
        &self,
        entity_index: I,
        entities_table_cache: &mut Vec<Option<Entity>>,
        children_query: &mut Query<(&Children,)>,
        name_query: &mut Query<(&Parent, &Name)>,
    ) -> Option<Entity> {
        if let Some(entity) = &entities_table_cache[entity_index.as_usize()] {
            Some(*entity)
        } else {
            let (parent_index, entity_name) = &self.entities[entity_index.as_usize()];

            // Use recursion to find the entity parent
            self.find_entity(
                *parent_index,
                entities_table_cache,
                children_query,
                name_query,
            )
            .and_then(|parent_entity| {
                if let Ok((children,)) = children_query.get(parent_entity) {
                    children
                        .iter()
                        .find(|entity| {
                            if let Ok((current_parent, name)) = name_query.get(**entity) {
                                // ! FIXME: Parent changes before the children update it self,
                                // ! to account for that we also must double check entity parent component it self
                                if current_parent.0 != parent_entity || name != entity_name {
                                    return false;
                                }

                                // Update cache
                                entities_table_cache[entity_index.as_usize()] = Some(**entity);
                                true
                            } else {
                                false
                            }
                        })
                        .copied()
                } else {
                    None
                }
            })
        }
    }

    /// Finds an entity given a reference to the (`World`)[bevy_ecs::World], see the example bellow
    /// how to proper call this function,
    ///
    /// ```rust
    /// let mut entities_table_cache = vec![];
    /// entities_table_cache.resize(clip.hierarchy.len(), None);
    /// // Assign the root entity as the first element
    /// entities_table_cache[0] = Some(root);
    ///
    /// let found_entity = named_hierarchy.find_entity_in_world(2, &mut entities_table_cache, &world);
    /// ```
    ///
    /// *NOTE* Keep in mind that you can have as many root as you want
    /// but each root must be manually find and inserted in the `entities_table_cache`
    /// before calling this function.
    pub fn find_entity_in_world(
        &self,
        entity_index: I,
        entities_table_cache: &mut Vec<Option<Entity>>,
        world: &World,
    ) -> Option<Entity> {
        if let Some(entity) = &entities_table_cache[entity_index.as_usize()] {
            Some(*entity)
        } else {
            let (parent_index, entity_name) = &self.entities[entity_index.as_usize()];

            // Use recursion to find the entity parent
            self.find_entity_in_world(*parent_index, entities_table_cache, world)
                .and_then(|parent_entity| {
                    if let Ok(children) = world.get::<Children>(parent_entity) {
                        children
                            .iter()
                            .find(|entity| {
                                if let Ok((current_parent, name)) =
                                    world.query_one::<(&Parent, &Name)>(**entity)
                                {
                                    // ! FIXME: Parent changes before the children update it self,
                                    // ! to account for that we also must double check entity parent component it self
                                    if current_parent.0 != parent_entity || name != entity_name {
                                        return false;
                                    }

                                    // Update cache
                                    entities_table_cache[entity_index.as_usize()] = Some(**entity);
                                    true
                                } else {
                                    false
                                }
                            })
                            .copied()
                    } else {
                        None
                    }
                })
        }
    }
}

mod private {
    use std::convert::TryFrom;

    /// Implemented by unsigned types
    pub trait Index: Sized + PartialEq + Copy + Clone {
        const MAX_VALUE: Self;

        fn as_usize(&self) -> usize;
        fn from_usize(index: usize) -> Self;
        fn from_usize_checked(index: usize) -> Self;
    }

    macro_rules! impl_index {
        ($t:ty) => {
            impl Index for $t {
                const MAX_VALUE: $t = <$t>::MAX;

                fn as_usize(&self) -> usize {
                    *self as usize
                }

                fn from_usize(index: usize) -> Self {
                    index as Self
                }

                fn from_usize_checked(index: usize) -> Self {
                    Self::try_from(index).expect(concat!(
                        "entities limit reached, indexed with ",
                        stringify!($t)
                    ))
                }
            }
        };
    }

    impl_index!(u8);
    impl_index!(u16);
    impl_index!(u32);
    impl_index!(u64);
    impl_index!(usize);
}
