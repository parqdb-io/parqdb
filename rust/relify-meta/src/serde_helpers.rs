use std::collections::BTreeMap;
use std::fmt;
use std::marker::PhantomData;

use serde::de::{Error as _, MapAccess, Visitor};
use serde::{Deserialize, Deserializer, Serializer};
use uuid::Uuid;

pub(crate) fn deserialize_unique_map<'de, D, K, V>(
    deserializer: D,
) -> std::result::Result<BTreeMap<K, V>, D::Error>
where
    D: Deserializer<'de>,
    K: Deserialize<'de> + Ord,
    V: Deserialize<'de>,
{
    struct UniqueMapVisitor<K, V>(PhantomData<(K, V)>);

    impl<'de, K, V> Visitor<'de> for UniqueMapVisitor<K, V>
    where
        K: Deserialize<'de> + Ord,
        V: Deserialize<'de>,
    {
        type Value = BTreeMap<K, V>;

        fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("a map with unique keys")
        }

        fn visit_map<M>(self, mut access: M) -> std::result::Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut values = BTreeMap::new();
            while let Some((key, value)) = access.next_entry()? {
                if values.insert(key, value).is_some() {
                    return Err(M::Error::custom("duplicate map key"));
                }
            }
            Ok(values)
        }
    }

    deserializer.deserialize_map(UniqueMapVisitor(PhantomData))
}

pub(crate) mod lowercase_uuid {
    use super::*;

    pub fn serialize<S>(value: &Uuid, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&value.to_string())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> std::result::Result<Uuid, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let parsed = Uuid::parse_str(&raw).map_err(D::Error::custom)?;
        if parsed.to_string() != raw {
            return Err(D::Error::custom(
                "UUID must use lowercase 8-4-4-4-12 hexadecimal form",
            ));
        }
        Ok(parsed)
    }
}
