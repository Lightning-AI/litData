package config

import (
	"fmt"
	"strconv"

	"github.com/Lightning-AI/litData/simulator/units"
	"gopkg.in/yaml.v3"
)

// ByteSize unmarshals 64MiB / 8GiB / integer bytes.
type ByteSize int64

func (b ByteSize) Int64() int64 { return int64(b) }

func (b *ByteSize) UnmarshalYAML(n *yaml.Node) error {
	if n.Kind != yaml.ScalarNode {
		return fmt.Errorf("byte size: want scalar")
	}
	if n.Tag == "!!int" || n.Tag == "!!float" {
		v, err := strconv.ParseInt(n.Value, 10, 64)
		if err != nil {
			f, err2 := strconv.ParseFloat(n.Value, 64)
			if err2 != nil {
				return err
			}
			*b = ByteSize(int64(f))
			return nil
		}
		*b = ByteSize(v)
		return nil
	}
	v, err := units.ParseBytes(n.Value)
	if err != nil {
		return err
	}
	*b = ByteSize(v)
	return nil
}

func (b ByteSize) MarshalYAML() (any, error) {
	return units.FormatBytes(int64(b)), nil
}

// Bandwidth unmarshals 1Gbps / 100MBps / integer bytes/s.
type Bandwidth int64

func (b Bandwidth) Int64() int64 { return int64(b) }

func (b *Bandwidth) UnmarshalYAML(n *yaml.Node) error {
	if n.Kind != yaml.ScalarNode {
		return fmt.Errorf("bandwidth: want scalar")
	}
	if n.Tag == "!!int" {
		v, err := strconv.ParseInt(n.Value, 10, 64)
		if err != nil {
			return err
		}
		*b = Bandwidth(v)
		return nil
	}
	v, err := units.ParseBandwidth(n.Value)
	if err != nil {
		return err
	}
	*b = Bandwidth(v)
	return nil
}
