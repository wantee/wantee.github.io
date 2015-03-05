module Jekyll
  module Url2FilenameFilter
    def url2filename(url)
      url = url.to_s.gsub(/\//, '-')
      url = url.sub(/^[^0-9]*-/, '')
      url = url.sub(/-$/, '')
    end
  end
end

Liquid::Template.register_filter(Jekyll::Url2FilenameFilter)

